"""Cut the subject out of a reference photo, and frame it the way the
generator expects.

Owned by LumenGen rather than delegated, because the trellis2 we now build on
never instantiates a background remover: inside ComfyUI that is a separate
node, so `preprocess_image` is off by default and its `rembg_model` stays None.
Feeding an uncut photo straight in makes the model try to reconstruct the
background as geometry - which is exactly how an eyeball on a black backdrop
came out as a flattened ovoid.

BiRefNet-HR is used rather than the RMBG-2.0 the stock config names: same
lineage, MIT rather than a non-commercial licence, no gated download, and
visibly better on thin structures - it keeps the gaps in filigree and the
separation between eyelashes.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("localmesh_engine.matting")

MODEL_ID = "ZhengPeng7/BiRefNet_HR"

#: The generator is trained on subjects that fill the frame. Padding is the
#: margin left around the bounding box, as a fraction of its largest side.
#:
#: PISTE FERMEE LE 26 AOUT 2026, avec la mesure. Le pretraitement de TRELLIS
#: lui-meme ne laisse AUCUNE marge (`size = int(size * 1)` dans
#: `preprocess_image`), et on pouvait croire que ces 4 % eloignaient le sujet
#: de la distribution d entrainement. Essaye a graine fixe sur deux sujets,
#: c est l inverse : sans marge, le personnage perd un oeil — il ne reste
#: qu un disque blanc — et la cassette rend ses bobines en bruit colore, avec
#: le texte imprime illisible. Avec 4 %, les deux yeux sont la et le texte se
#: lit. Ne pas y revenir sans une meilleure mesure que celle-la.
_PADDING = 0.04
#: Working resolution of the matte network.
_MATTE_SIZE = 1024
#: Anything above this is downscaled first: a 12 MP phone photo costs seconds
#: of matting for detail the 1024-wide network cannot use.
_MAX_INPUT = 2048


class Matter:
    """Lazily-loaded background remover. One instance per process."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        from transformers import AutoModelForImageSegmentation

        log.info("loading matting model %s", self.model_id)
        model = AutoModelForImageSegmentation.from_pretrained(
            self.model_id, trust_remote_code=True)
        model.eval().to("cuda")
        self._model = model
        return model

    def unload(self):
        self._model = None
        torch.cuda.empty_cache()

    # -- matte ---------------------------------------------------------------

    def alpha(self, image: Image.Image) -> Image.Image:
        """Return the subject's alpha as an L-mode image at the input size."""
        from torchvision import transforms

        model = self.load()
        dtype = next(model.parameters()).dtype  # checkpoints ship fp16

        tf = transforms.Compose([
            transforms.Resize((_MATTE_SIZE, _MATTE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        batch = tf(image.convert("RGB")).unsqueeze(0).to("cuda", dtype=dtype)
        with torch.no_grad():
            pred = model(batch)[-1].sigmoid().float().cpu()[0].squeeze()
        return transforms.ToPILImage()(pred).resize(image.size, Image.BILINEAR)

    # -- full preprocessing --------------------------------------------------

    def prepare(self, image: Image.Image) -> Image.Image:
        """Matte, crop to the subject, square up, and premultiply.

        Mirrors what TRELLIS' own preprocessing did, because the generator was
        trained on images framed that way: subject centred, filling the frame,
        black where there is nothing.
        """
        return self.preparer_avec_geometrie(image)[0]

    def _detourer(self, image: Image.Image):
        """Detourer, et rendre de quoi recadrer ensuite — sans recadrer.

        Separe de `preparer_avec_geometrie` pour que le LOT (`preparer_lot`)
        puisse decider d'un cote de recadrage COMMUN aux quatre vues apres les
        avoir toutes vues. Rend `(travail, alpha, boite_sujet, taille_source)`,
        ou `boite_sujet` vaut None quand rien n'a ete trouve.
        """
        taille_source = tuple(image.size)
        if max(image.size) > _MAX_INPUT:
            scale = _MAX_INPUT / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.LANCZOS)

        alpha = None
        if image.mode == "RGBA":
            existing = np.array(image)[:, :, 3]
            if not np.all(existing == 255):
                alpha = Image.fromarray(existing)
        if alpha is None:
            alpha = self.alpha(image)

        a = np.asarray(alpha, dtype=np.float32) / 255.0
        solide = np.argwhere(a > 0.8)
        if solide.size == 0:
            return image, a, None, taille_source
        y0, x0 = solide.min(axis=0)
        y1, x1 = solide.max(axis=0)
        return image, a, (int(x0), int(y0), int(x1), int(y1)), taille_source

    @staticmethod
    def _recadrer(image, a, sujet, taille_source, cote):
        """Le recadrage carre, autour du sujet, au cote demande.

        LE NOIR AUTOUR N'EST PAS UN ACCIDENT. L'image detouree est deja le
        sujet sur fond noir ; une boite qui deborde de la photo rend donc du
        noir, exactement la meme matiere que le fond. C'est ce qui permet de
        donner de l'air a un sujet qui touchait le bord du cadre.
        """
        x0, y0, x1, y1 = sujet
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        boite = (int(cx - cote / 2), int(cy - cote / 2),
                 int(cx + cote / 2), int(cy + cote / 2))
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        cut = Image.fromarray(
            ((rgb * a[..., None]) * 255).astype(np.uint8)).crop(boite)
        log.info("matted subject: %dx%d crop, covers %.1f%% of the frame",
                 cut.width, cut.height, 100 * float((a > 0.5).mean()))
        return cut, _geometrie_de_preparation(
            taille_source, image.size, boite, sujet)

    def preparer_lot(self, images: dict):
        """Detourer QUATRE vues du meme objet a la MEME echelle.

        LE DEFAUT QUE CA REPARE, ET IL EST VISIBLE. Le recadrage d'une vue
        seule prend pour cote la plus grande dimension du sujet. Sur un objet
        plus haut que large sous tous les angles — une tete, un samourai — la
        plus grande dimension EST la hauteur, les quatre cadres sont donc deja
        a la meme echelle et ce lot ne change rigoureusement rien.
        Mesure : 0,0 % d'ecart avant comme apres.
        
        Sur un objet ALLONGE, non. Une moto vue de profil est cadree sur sa
        LONGUEUR : sa hauteur ne remplit alors que 65 % du cadre, contre 96 %
        vue de face. Les quatre vues arrivent au modele a des echelles qui
        different de 38 %, la fusion croit que la moto est aussi large que
        longue, et elle l'epaissit. Mesure du 10 septembre 2026 : moto 37,9 %
        d'ecart, cassette 39,7 %, samourai et tete 0,0 %.

        CE QUI SERT DE REGLE : une rotation autour de la verticale ne change
        pas la hauteur d'un objet. C'est donc la HAUTEUR, et elle seule, qui
        dit si deux vues sont a la meme echelle. On prend un cote commun
        proportionnel a la hauteur du sujet dans chaque vue, avec le plus
        petit facteur qui fasse encore tenir la vue la plus large.

        CE QUE CA COUTE, ET LA NOTE PLUS BAS L'AVAIT PREVU : sur un objet
        allonge le sujet ne remplit plus le cadre a 96 % mais a 64 %, et les
        modeles ont ete entrainis sur des sujets qui le remplissent. C'est le
        prix a payer, et c'est bien moins cher que le transport vers le cadre
        d'origine, qui reduisait le sujet a une fraction bien plus petite.
        C'est la « piste propre » que la note appelait de ses voeux : l'union
        des recadrages, ou le sujet remplit ENCORE le cadre.
        """
        detoures = {role: self._detourer(im) for role, im in images.items()}
        facteur = 1.0
        for _im, _a, sujet, _ts in detoures.values():
            if not sujet:
                continue
            x0, y0, x1, y1 = sujet
            h = max(y1 - y0, 1)
            facteur = max(facteur, max(x1 - x0, y1 - y0) / h)
        sortie = {}
        for role, (im, a, sujet, ts) in detoures.items():
            if not sujet:
                log.warning("matting found no subject; using the image as-is")
                sortie[role] = (im.convert("RGB"), _geometrie_de_preparation(
                    ts, im.size, (0, 0, im.width, im.height), None))
                continue
            cote = int(max(sujet[3] - sujet[1], 1) * facteur * (1 + _PADDING))
            sortie[role] = self._recadrer(im, a, sujet, ts, cote)
        if facteur > 1.001:
            log.info("cadre commun aux vues : facteur %.3f "
                     "(sujet allonge, les echelles sont remises d'accord)",
                     facteur)
        return sortie

    def preparer_avec_geometrie(self, image: Image.Image):
        """La meme preparation, et de quoi revenir au cadre d'origine.

        POURQUOI CE SECOND RETOUR EXISTE. Le recadrage ci-dessous serre CHAQUE
        photo sur son propre sujet. Sur une seule photo c'est exactement ce
        qu'il faut : le modele a ete entraine sur des sujets centres qui
        remplissent le cadre. Sur quatre photos du meme objet, c'est un piege,
        parce que la silhouette change d'une vue a l'autre : un sabre qui sort
        du cadre de face et pas de dos fait deux recadrages differents, donc
        deux echelles differentes, donc un meme point de l'objet qui ne tombe
        plus au meme endroit selon la vue.

        Le multivue projette les quatre vues sur une grille commune. Si les
        quatre cadres ne s'accordent pas, la correspondance est fausse et la
        fusion construit un objet qui n'existe pas. MESURE SUR LE SAMOURAI :
        un visage sculpte a l'avant ET a l'arriere.

        Ce dictionnaire porte de quoi ramener une projection faite dans le
        cadre d'origine vers le cadre recadre. Le cadre d'origine est commun
        aux quatre vues quand elles viennent du meme appareil, ce qui est le
        cas d'un tour d'objet ; l'erreur de cadrage devient alors la meme pour
        les quatre, au lieu d'etre differente pour chacune.
        """
        taille_source = tuple(image.size)
        if max(image.size) > _MAX_INPUT:
            scale = _MAX_INPUT / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.LANCZOS)

        # An alpha channel that is already meaningful is trusted as-is: it is
        # better than anything we would re-derive, and re-matting a cutout
        # tends to eat its edges.
        alpha: Optional[Image.Image] = None
        if image.mode == "RGBA":
            existing = np.array(image)[:, :, 3]
            if not np.all(existing == 255):
                alpha = Image.fromarray(existing)

        if alpha is None:
            alpha = self.alpha(image)

        a = np.asarray(alpha, dtype=np.float32) / 255.0
        solid = np.argwhere(a > 0.8)
        if solid.size == 0:  # nothing found; hand back the original framing
            log.warning("matting found no subject; using the image as-is")
            return image.convert("RGB"), _geometrie_de_preparation(
                taille_source, image.size, (0, 0, image.width, image.height),
                None)

        y0, x0 = solid.min(axis=0)
        y1, x1 = solid.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = int(max(x1 - x0, y1 - y0) * (1 + _PADDING))
        box = (int(cx - side / 2), int(cy - side / 2),
               int(cx + side / 2), int(cy + side / 2))

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        cut = Image.fromarray(
            ((rgb * a[..., None]) * 255).astype(np.uint8)).crop(box)
        log.info("matted subject: %dx%d crop, covers %.1f%% of the frame",
                 cut.width, cut.height, 100 * float((a > 0.5).mean()))
        return cut, _geometrie_de_preparation(
            taille_source, image.size, box,
            (int(x0), int(y0), int(x1), int(y1)))


#: UNE PISTE FERMEE PAR LA MESURE, ET QUI NE DOIT PAS SE ROUVRIR.
#:
#: Les quatre vues du buste ne s'accordent pas sur la taille de l'objet : 8,7
#: pour cent d'ecart de hauteur dans les photos d'origine. On a donc ecrit un
#: detourage commun, qui donnait aux quatre vues la meme echelle et la meme
#: hauteur, en pensant que ce decalage expliquait les deux nez.
#:
#: IL N'EXPLIQUAIT RIEN. Mesure faite sur les images REELLEMENT envoyees au
#: modele : le recadrage carre ci-dessous egalise deja tout. L'objet occupe
#: 96,4 % de la hauteur du cadre et son centre tombe a 50,0 %, dans les quatre
#: vues, avec ou sans cadre commun. L'ecart de 8,7 % vit dans le cadre
#: D'ORIGINE et meurt au recadrage.
#:
#: Ce qui reste, et qu'aucun recadrage ne repare : les quatre images ne
#: s'accordent pas sur les PROPORTIONS INTERNES de l'objet — ou est le nez dans
#: le visage. C'est un defaut des photos, pas du cadrage.

def _geometrie_de_preparation(taille_source, taille_travail, boite,
                              sujet=None) -> dict:
    """Porter un point du cadre de travail vers le cadre recadre.

    Les deux cadres sont exprimes en coordonnees normalisees de centre de
    pixel, la convention de `grid_sample(align_corners=False)` : un point vaut
    `(pixel + 0.5) / cote * 2 - 1`. Redimensionner l'image entiere ne change
    pas ces coordonnees, donc seul le recadrage compte, et il se resume a une
    echelle et un decalage.

    La boite peut sortir de l'image : le recadrage garde alors du noir autour,
    et les coordonnees restent justes. Une boite vide rend `valide` faux
    plutot qu'une division par zero.
    """
    largeur, hauteur = taille_travail
    gauche, haut, droite, bas = boite
    # LA BOITE DU SUJET, PAS LE RECADRAGE. Le recadrage est CARRE : son cote
    # vaut la plus grande dimension du sujet, donc sa hauteur ne dit pas la
    # hauteur du sujet des qu'il est plus large que haut. Or c'est la hauteur
    # qui doit rester la meme quand on tourne autour d'un objet, et c'est donc
    # elle qui dit si quatre images montrent bien le meme objet.
    sujet_boite = list(sujet) if sujet else None
    touche_le_bord = bool(
        sujet and (sujet[1] <= 0 or sujet[3] >= hauteur - 1))
    lc, hc = droite - gauche, bas - haut
    valide = lc > 0 and hc > 0
    return {
        "version": 1,
        "convention": "centres_de_pixel_normalises_align_corners_false",
        "source_size": list(taille_source),
        "working_size": list(taille_travail),
        "crop_box": list(boite),
        "prepared_size": [lc, hc],
        "sujet_boite": sujet_boite,
        "sujet_hauteur_relative": ((sujet[3] - sujet[1] + 1) / hauteur
                                   if sujet and hauteur else None),
        "sujet_touche_le_bord": touche_le_bord,
        "valid": valide,
        "normalized_scale": [largeur / lc, hauteur / hc] if valide else None,
        "normalized_offset": [(largeur - 2 * gauche) / lc - 1,
                              (hauteur - 2 * haut) / hc - 1] if valide else None,
    }


_matter: Optional[Matter] = None


def matter() -> Matter:
    global _matter
    if _matter is None:
        _matter = Matter()
    return _matter


def est_charge() -> bool:
    """Le detourage occupe-t-il la carte ?"""
    return _matter is not None and _matter._model is not None

def liberer() -> None:
    """Rendre la mémoire vidéo du détourage.

    Ses voisines — mogenormals, idarb, supermat, inpaint — sont rendues
    après chaque conversion sur une carte à la peine. Celle-ci ne l'était
    pas, faute d'une fonction de module à appeler : elle gardait son
    gigaoctet pour toute la session. C'est exactement la forme du « la
    première génération passe, la deuxième sature ».
    """
    if _matter is not None:
        _matter.unload()
