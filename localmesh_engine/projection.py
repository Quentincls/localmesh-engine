"""La projection des photos : coller les VRAIS pixels sur la texture.

La texture que rend TRELLIS.2 est DESSINÉE par le modèle à partir des photos.
Elle est fidèle en gros et fausse dans le détail : un texte imprimé devient du
charabia, un motif fin se brouille. Ce module reprend chaque texel que l'une
des photos voit, et y recopie le pixel de la photo. Là, rien n'est inventé :
le texte redevient EXACT, pas « plus net ».

Là où aucune photo ne voit la surface — le dessous, l'intérieur, ce qui est
caché — la texture dessinée reste, et sa couleur est accordée localement à
celle des photos voisines.

TOUT CE QUI SUIT A ÉTÉ MESURÉ sur un banc qui texture le MÊME maillage figé
de plusieurs façons, les 22 et 23 septembre 2026, sur une cassette (texte
imprimé), un samouraï et une moto, et confirmé à l'œil à chaque étape.

CE QUE LA PROJECTION NE PEUT PAS FAIRE. Une photo juste collée sur une forme
fausse dédouble : sur la moto, le silencieux 3D sort en cône là où la photo
montre un tube, et le vrai pot atterrit en partie sur le bras oscillant. Aucun
réglage de texture ne corrige une géométrie. C'est pourquoi la projection se
DÉSACTIVE (`projection_texture=False`) : on retombe alors sur la texture
dessinée, floue mais jamais dédoublée.

Licences : rien de neuf. OpenCV n'est pas utilisé ici ; `uvraster` est le
rastériseur du moteur (pas de nvdiffrast), `matting` et `multivue.cotes` sont
déjà là.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from . import uvraster
from .multivue import cameras as cam

log = logging.getLogger(__name__)

#: Les rôles des photos et leur angle NOMINAL, remplacé par l'angle mesuré par
#: Depth Anything 3 quand il répond (sur le samouraï : 52,7° au lieu de 90).
ROLES = {"face": 0.0, "droite": 90.0, "dos": 180.0, "gauche": 270.0}
VERS_ROLE = {"face": "front", "droite": "right", "dos": "back", "gauche": "left"}


@dataclass
class Reglages:
    """Les réglages retenus. Chacun a été balayé ; la valeur est celle qui a
    gagné À L'ŒIL sur les trois sujets, pas seulement au chiffre."""

    #: Sous ce cosinus (≈ 80°), la photo rase la surface et le motif s'étire.
    #: 0,40 au départ ; 0,18 retenu : relâcher seul ne rend qu'un tiers de ce
    #: qu'il refuse, car une surface vue de profil est AUSSI près de sa propre
    #: silhouette, que la falaise refuse déjà.
    cos_mini: float = 0.18
    #: Érosion du détourage, en pixels de la photo préparée. Retire la bande
    #: d'alpha douce : le détourage rend du PRÉMULTIPLIÉ SUR NOIR, et chaque
    #: structure fine arrivait bordée d'un halo noir — c'était le « brun sale ».
    erosion: int = 3
    #: Marge autour des falaises de profondeur, en texels de l'atlas (voir
    #: `_falaises`). 9 au départ, 5 une fois la caméra calée à zéro pixel.
    marge_falaise: int = 5
    #: Un saut de profondeur au moins égal à ceci, en fraction de l'objet,
    #: entre deux pixels voisins : c'est un bord de pièce.
    saut_falaise: float = 0.012
    #: Un texel est « bien lui » si l'UV rendue à son pixel en est à moins de
    #: ceci, en texels de l'atlas.
    tolerance_uv_texels: float = 1.5
    #: Accorder la couleur de la texture dessinée à celle des photos, là où
    #: aucune photo ne voit. Mesuré sur des blocs de photo CACHÉS : Delta E
    #: médian 17,8 -> 15,4 (moto), 13,7 -> 12,8 (cassette), 19,4 -> 16,9
    #: (samouraï). Gain modeste : l'écart restant est surtout du CONTENU.
    harmonie: bool = True
    appui_harmonie: float = 5.0
    #: Part de la correction de TEINTE gardée (la luminosité l'est en entier),
    #: et son plafond, en logarithme du rapport par canal. Sans eux, un écart
    #: lavande contre gris devenait du vert d'eau ailleurs sur l'objet, et
    #: un reflet chaud de la photo une tache jaune là où aucune photo ne voit.
    harmonie_teinte: float = 0.4
    harmonie_teinte_max: float = 0.08
    #: À accord de silhouette égal, l'appareil de niveau est préféré.
    pente_niveau: float = 0.003
    #: Résolutions du calage : on balaie en 384, on polit en 768.
    r_cal: int = 384
    r_fin: int = 768
    #: De quel côté est la face : l'écart d'accord de silhouette, moyen par
    #: vue, puis l'écart de ressemblance de la texture dessinée, au-delà
    #: desquels on tranche. En deçà, la face est devant (+Z).
    ecart_sens_silhouette: float = 0.03
    ecart_sens_apparence: float = 0.05
    #: Une photo SEULE n'est pas toujours prise de face. On cherche son azimut
    #: jusqu'à ceci de part et d'autre de la face, en degrés ; 0 le fige.
    #: À accord égal, la face franche est préférée, comme l'horizontale.
    #: TOUT LE TOUR depuis le 24 septembre 2026 : le modèle sort parfois
    #: tourné d'un quart de tour (un téléphone vu de face, fabriqué face à
    #: 90°). À ±60° la recherche butait sur sa borne (51°, accord 0,77) et
    #: collait le cadran en biais ; sur le tour entier elle trouve 87° (0,88).
    #: Les sept sujets à une photo du banc gardent exactement leur angle : la
    #: pente ci-dessous écarte l'image miroir (bâtiment, lampadaire, pot rond
    #: à 180°), qui colle parfois un rien mieux. Coût : 6 à 17 s de calage.
    azimut_libre: float = 180.0
    pente_face: float = 0.002
    #: Le ROULIS de l'appareil, en degrés : jusqu'où on le cherche avec une
    #: photo seule, et avec plusieurs (où la forme a été générée d'aplomb).
    roulis_libre: float = 20.0
    roulis_libre_mv: float = 3.0
    #: Le recalage fin par le relief (`relief.py`) : la photo est lue là où
    #: son relief tombe sur celui du modèle. Sans les poids de MoGe-2 sur le
    #: disque, il est sauté et la caméra rigide reste seule.
    #:
    #: COUPÉ PAR DÉFAUT (24 sept 2026). Jury à l'aveugle sur 14 sujets, 57 zooms
    #: A/B et 3 regards : sans 51, avec 28, égal 92. Il recale parfois un détail
    #: (le clignotant de la moto), mais il tord aussi des bords droits en S (le
    #: feu tricolore) ; il coûte ~6 s par projection et 1,3 Go de poids que les
    #: clients n'ont pas. Le code reste, pour un meilleur modèle de relief.
    recalage_relief: bool = False
    #: Le mélange photo / dessin. « doux » : la part de photo vaut poids^0,35,
    #: si bien qu'aux angles rasants le dessin transparaît sous la photo. « franc » :
    #: la photo couvre pleinement au-delà de `franc_seuil`, ses petits trous
    #: (jusqu'à `trous_texels` de rayon) sont bouchés par ses voisins, et le fondu
    #: ne se fait que sur `bord_texels` au bord réel de la zone photographiée.
    #: Les GOUTTIÈRES entre îles UV, remplies APRÈS la projection depuis la
    #: couleur finale des îles, sur tant de texels. Elles l'étaient seulement
    #: avant (cv2, dans `postprocess_mesh`) : elles gardaient la couleur
    #: DESSINÉE, et une visionneuse à mipmaps montrait un fin trait le long
    #: de chaque découpe (14 à 20 ΔE mesurés, contre moins de 2).
    gouttieres_texels: int = 8
    #: LE FONDU AU BORD DE LA PHOTO, en part de l'objet (étendue 1). Près de la
    #: limite de ce que la photo voit, le dessin de l'IA reprend la main en
    #: douceur au lieu d'un trait net (le combiné du téléphone de Quentin, le
    #: 24 septembre 2026). La distance est prise SUR LA SURFACE, en 3D, à la
    #: plus proche zone non peinte du même côté : elle ne voit pas les coutures
    #: de l'atlas. 0 le coupe.
    #:
    #: MESURÉ le 24 septembre 2026 (13 sujets, jury aveugle à trois versions,
    #: 41 zooms x 3 regards). En UNE photo, 1,5 % gagne : la photo y rejoint
    #: un grand dessin de l'IA, et la jonction nette se voit (lampadaire, banc,
    #: couteau, statue ; seul le bâtiment préfère sans). À QUATRE photos, sans
    #: fondu gagne nettement (+45 contre +11) : presque tout est photo, les
    #: zones sans photo ne sont que des bandes fines, et estomper autour ne
    #: fait que perdre du détail. 3 % efface trop (pire 68 fois sur 120).
    fondu: float = 0.015
    fondu_multivue: float = 0.0
    melange: str = "doux"
    franc_seuil: float = 0.2
    trous_texels: int = 3
    bord_texels: int = 6


@dataclass
class Rapport:
    """Ce que la projection a fait, pour le journal et pour le client."""
    vues: dict = field(default_factory=dict)
    elevation: float = 0.0
    couverture: float = 0.0
    azimuts_mesures: bool = False
    #: 0 : la face du modèle regarde +Z ; 180 : elle regarde -Z.
    sens_face: float = 0.0
    sens_decide_par: str = ""
    #: Vrai si les photos « droite » et « gauche » ont été prises de l'autre
    #: côté que leur emplacement ne le dit, et projetées en conséquence.
    cotes_echanges: bool = False
    #: Pour une photo seule : de combien, en degrés, elle a été prise à
    #: côté de la face.
    azimut_photo: float = 0.0
    #: Part de la silhouette de chaque vue que le recalage par le relief a
    #: déplacée, en pour cent. Vide s'il n'a pas tourné.
    recalage_relief: dict = field(default_factory=dict)
    duree: float = 0.0
    duree_calage: float = 0.0


# ------------------------------------------------------------- couleurs ----
def _vers_lineaire(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _vers_srgb(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# --------------------------------------------------------------- caméra ----
def _proj(p, yaw, focal, ecran, elev=0.0, ech=1.0, dx=0.0, dy=0.0, roul=0.0):
    """La caméra du moteur (`multivue.cameras`), plus l'ÉLÉVATION et un
    RECENTRAGE. `dx`, `dy` sont en fraction d'écran, pour valoir à toute
    résolution.

    L'élévation n'est pas un raffinement : projeter une photo prise à −16°
    comme si elle l'était à 0° décale l'image différemment selon la
    PROFONDEUR, et l'avant et l'arrière de l'objet glissent l'un sur l'autre.
    C'était le premier dédoublement vu sur la moto.

    `roul` fait tourner l'image autour de l'axe optique. Sans lui, un couteau
    photographié penché de 8° n'avait que 0,43 d'accord de silhouette : sa
    lame tombait de plus en plus à côté vers la pointe.
    """
    p = p * ech
    cy, sy = math.cos(yaw), math.sin(yaw)
    x1 = cy * p[:, 0] - sy * p[:, 2]
    z1 = sy * p[:, 0] + cy * p[:, 2]
    ce, se = math.cos(elev), math.sin(elev)
    y2 = ce * p[:, 1] - se * z1
    z2 = se * p[:, 1] + ce * z1
    d = cam.DISTANCE - z2
    xc = focal * x1 / (d + 1e-8)
    yc = -focal * y2 / (d + 1e-8)
    if roul:
        cr, sr = math.cos(roul), math.sin(roul)
        xc, yc = cr * xc - sr * yc, sr * xc + cr * yc
    x = xc + ecran / 2 + dx * ecran
    y = yc + ecran / 2 + dy * ecran
    return x, y, d


def _clip(x, y, d, ecran):
    """Coordonnées homogènes NON divisées : `uvraster` corrige alors la
    perspective en interpolant les attributs."""
    ndc_x = x / ecran * 2 - 1
    ndc_y = -(y / ecran * 2 - 1)
    ndc_z = (d - cam.DISTANCE + 1.0) * 0.5
    return torch.stack([ndc_x * d, ndc_y * d, ndc_z * d, d], 1)


def _focale(ecran):
    return ecran / (2 * math.tan(cam.FOV / 2))


def _sil(V, F, foc, ecran, yaw, elev, ech, dx, dy, tournees=True, roul=0.0):
    """La silhouette rendue. Ne dessiner que les faces tournées vers
    l'appareil divise presque le temps par deux, pour un accord de 99,6 à
    99,9 % avec la silhouette complète."""
    x, y, d = _proj(V, yaw, foc, ecran, elev, ech, dx, dy, roul)
    if tournees:
        a = torch.stack([x, y], 1)[F]
        aire = ((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0]))
        F = F[aire > 0]
    _, cov = uvraster.camera(_clip(x, y, d, ecran), F, ecran, attributes=())
    return cov.reshape(ecran, ecran) > 0


def _boite(m):
    ys = m.any(1).nonzero()
    xs = m.any(0).nonzero()
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _iou(a, b):
    return float((a & b).sum()) / max(float((a | b).sum()), 1.)


def _par_boite(V, F, photo, foc, ecran, yaw, elev, roul=0.0):
    """Taille et centrage tirés des boîtes englobantes : deux silhouettes au
    lieu d'un balayage de quatre-vingts."""
    bp = _boite(photo)
    bm = _boite(_sil(V, F, foc, ecran, yaw, elev, 1., 0., 0., roul=roul))
    if bp is None or bm is None:
        return 0., 1., 0., 0.
    wm, hm = bm[2] - bm[0] + 1, bm[3] - bm[1] + 1
    wp, hp = bp[2] - bp[0] + 1, bp[3] - bp[1] + 1
    e = math.sqrt((wp * hp) / max(wm * hm, 1.))
    c = ecran / 2
    dx = (((bp[0] + bp[2]) / 2 - c) - e * ((bm[0] + bm[2]) / 2 - c)) / ecran
    dy = (((bp[1] + bp[3]) / 2 - c) - e * ((bm[1] + bm[3]) / 2 - c)) / ecran
    return (_iou(_sil(V, F, foc, ecran, yaw, elev, e, dx, dy, roul=roul), photo),
            e, dx, dy)


def _affiner(V, F, photo, foc, ecran, yaw, elev, depart, pas, mini):
    """Recherche par motifs sur (taille, dx, dy), et le roulis en quatrième
    si `depart` en a quatre."""
    def f(e, dx, dy, roul=0.0):
        return _iou(_sil(V, F, foc, ecran, yaw, elev, e, dx, dy, roul=roul), photo)
    best = (f(*depart),) + tuple(depart)
    pas = list(pas)
    n = 1
    while any(p > m for p, m in zip(pas, mini)) and n < 160:
        mieux = False
        for k in range(len(pas)):
            for sg in (1., -1.):
                cand = list(best[1:])
                cand[k] += sg * pas[k]
                v = f(*cand)
                n += 1
                if v > best[0]:
                    best, mieux = (v,) + tuple(cand), True
                    break
            if mieux:
                break
        if not mieux:
            pas = [p / 2 for p in pas]
    return best


def _ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def _silhouettes(V, F, alphas, yaws, rg: Reglages, roles):
    """L'accord de silhouette moyen de `roles`, chacun au mieux de trois
    élévations : le sens se décide avant que l'élévation soit calée."""
    foc = _focale(rg.r_cal)
    return sum(max(_par_boite(V, F, alphas[r], foc, rg.r_cal, yaws[r],
                              math.radians(e))[0] for e in (-8.0, 0.0, 8.0))
               for r in roles) / max(len(roles), 1)


def _ressemblance(V, F, UV, T, alphas, photos_cal, yaws, rg: Reglages, roles):
    """La corrélation moyenne entre la texture dessinée, rendue depuis chaque
    caméra de `roles`, et la luminance de la photo."""
    foc = _focale(rg.r_cal)
    total, n = 0.0, 0
    for r in roles:
        _v, e, dx, dy = _par_boite(V, F, alphas[r], foc, rg.r_cal, yaws[r], 0.0)
        x, y, d = _proj(V, yaws[r], foc, rg.r_cal, 0.0, e, dx, dy)
        (uvb,), cov = uvraster.camera(_clip(x, y, d, rg.r_cal), F, rg.r_cal,
                                      attributes=(UV,))
        st = uvb.reshape(rg.r_cal, rg.r_cal, 2).clone()
        st[..., 1] = 1.0 - st[..., 1]
        rendu = uvraster.sample(T, st).reshape(rg.r_cal, rg.r_cal, 3).mean(2)
        m = (cov.reshape(rg.r_cal, rg.r_cal) > 0) & alphas[r]
        if int(m.sum()) > 100:
            total += _ncc(rendu[m], photos_cal[r][m])
            n += 1
    return total / max(n, 1)


def _sens_de_la_face(V, F, UV, T, alphas, photos_cal, yaws, rg: Reglages,
                     cotes_surs=True):
    """De quel côté du modèle est la face photographiée, devant (+Z) ou
    derrière (-Z), et les deux profils sont-ils rangés à l'envers ?
    Rend (0 ou pi, profils échangés, ce qui a tranché).

    POURQUOI CE N'EST PAS UNE CONVENTION. Mesuré le 23 septembre 2026 sur
    seize générations : le modèle sort le plus souvent sa face vers +Z, mais
    pas toujours. Un couteau généré ce jour-là regardait -Z, silhouette 0,11
    d'accord vue de devant contre 0,46 vue de derrière. Projeté du mauvais
    côté, le couteau a reçu sa photo sur le dos, de travers.

    POURQUOI LES PROFILS À PART. Tourner l'objet d'un demi-tour et échanger
    les deux profils donnent EXACTEMENT les mêmes silhouettes de côté. Un feu
    tricolore dont les photos « droite » et « gauche » étaient rangées à
    l'envers a été retourné en entier : ses profils collaient mieux, et les
    feux photographiés sont allés sur les trappes du dos. Quand les angles
    ne sont ni mesurés ni imposés (`cotes_surs` faux), les profils ne disent
    donc que « demi-tour OU profils échangés », et le sens se lit sur la
    face et le dos seuls.

    D'ABORD LA SILHOUETTE de la face et du dos, qui tranche dès que l'objet
    n'est pas symétrique. ENSUITE LA RESSEMBLANCE de la texture dessinée à
    ces photos : le modèle dessine la face d'après la photo et invente le
    dos, donc la face lui ressemble. Elle ne sert que si son écart est net.
    SINON, DEVANT : c'est le cas le plus fréquent.
    """
    cotes = [r for r in ("droite", "gauche") if r in alphas]
    face_dos = [r for r in ("face", "dos") if r in alphas]

    def vers(off, echange=False):
        return {r: y + off + (math.pi if echange and r in cotes else 0.0)
                for r, y in yaws.items()}

    # Les profils, quand on n'en est pas sûr : « tels quels » contre
    # « échangés », l'objet laissé tel qu'il est.
    profils = None
    if cotes and not cotes_surs:
        ecart = (_silhouettes(V, F, alphas, vers(0.0, True), rg, cotes)
                 - _silhouettes(V, F, alphas, vers(0.0), rg, cotes))
        if abs(ecart) > rg.ecart_sens_silhouette:
            profils = ecart > 0          # vrai : un demi-tour OU un échange
    roles = list(alphas) if cotes_surs else face_dos

    sens, par = None, "défaut"
    ecart = (_silhouettes(V, F, alphas, vers(0.0), rg, roles)
             - _silhouettes(V, F, alphas, vers(math.pi), rg, roles))
    if abs(ecart) > rg.ecart_sens_silhouette:
        sens, par = (0.0 if ecart > 0 else math.pi), "silhouette"
    else:
        ecart = (_ressemblance(V, F, UV, T, alphas, photos_cal, vers(0.0), rg,
                               face_dos)
                 - _ressemblance(V, F, UV, T, alphas, photos_cal, vers(math.pi),
                                 rg, face_dos))
        if abs(ecart) > rg.ecart_sens_apparence:
            sens, par = (0.0 if ecart > 0 else math.pi), "apparence"
    if sens is None:
        sens = 0.0
    echange = (profils != (sens != 0.0)) if profils is not None else False
    return sens, echange, par


def _azimut_de_la_photo(V, F, alpha, yaw, rg: Reglages):
    """D'où une photo SEULE a été prise : de combien à côté de la face, et
    de quelle hauteur. Rend (décalage d'azimut en radians, élévation en
    degrés), l'élévation n'étant qu'un point de départ pour `_caler`.

    POURQUOI. Une photo seule était supposée prise de face. Mesuré le
    23 septembre 2026 : un banc photographié de trois-quarts et d'en haut
    passe de 0,58 à 0,90 d'accord de silhouette quand on cherche aussi
    l'azimut (-34°, +13°), un couteau de 0,42 à 0,79 (-48°), une statue de
    0,79 à 0,88 (+26°, -17°). Projetée comme de face, la photo du banc
    peignait l'assise une seconde fois sur le dossier. Sur une photo
    vraiment de face (bâtiment, poupée, lampadaire), la recherche retombe
    à moins de 8° de zéro et l'accord ne bouge pas. Et le 24 septembre, un
    téléphone fabriqué face à 90° : voir `Reglages.azimut_libre`.

    Un balayage grossier, puis un second plus fin autour du meilleur.
    """
    foc = _focale(rg.r_cal)
    libre = rg.azimut_libre

    def note(az, el):
        v = _par_boite(V, F, alpha, foc, rg.r_cal, yaw + math.radians(az),
                       math.radians(el))[0]
        return v - rg.pente_face * abs(az) - rg.pente_niveau / 4 * abs(el)

    grille = [(az, el) for az in np.arange(-libre, libre + 0.1, 15.)
              for el in np.arange(-15., 45.1, 10.)]
    az0, el0 = max(grille, key=lambda c: note(*c))
    fine = [(az, el) for az in np.arange(az0 - 6., az0 + 6.1, 3.)
            for el in np.arange(el0 - 6., el0 + 6.1, 3.)]
    az, el = max(fine, key=lambda c: note(*c))
    return math.radians(float(az)), float(el)


def _axe(m):
    """L'angle de l'axe principal d'un masque (radians, dans le repère des
    pixels) et l'allongement : rapport de ses deux valeurs propres."""
    ys, xs = torch.nonzero(m, as_tuple=True)
    if xs.numel() < 50:
        return 0.0, 1.0
    x = xs.float() - xs.float().mean()
    y = ys.float() - ys.float().mean()
    cxx, cyy, cxy = float((x * x).mean()), float((y * y).mean()), float((x * y).mean())
    t = math.sqrt(max(((cxx - cyy) / 2) ** 2 + cxy ** 2, 0.0))
    l1, l2 = (cxx + cyy) / 2 + t, (cxx + cyy) / 2 - t
    return 0.5 * math.atan2(2 * cxy, cxx - cyy), l1 / max(l2, 1e-6)


def _roulis_de_depart(V, F, photo, foc, ecran, yaw, elev, borne):
    """Le roulis qui aligne l'axe principal de la silhouette sur celui de la
    photo, quand l'objet est assez allongé pour avoir un axe. Borné."""
    ap, lp = _axe(photo)
    am, lm = _axe(_sil(V, F, foc, ecran, yaw, elev, 1., 0., 0.))
    if min(lp, lm) < 1.5:
        return 0.0
    r = (ap - am + math.pi / 2) % math.pi - math.pi / 2
    return max(-borne, min(borne, r))


def _caler(V, F, alphas, alphas_fin, yaws, rg: Reglages, elev_depart=None):
    """UN SEUL APPAREIL POUR TOUTES LES VUES : une élévation partagée, taille
    et centrage par vue.

    Pourquoi partagée : laissée libre par vue, la moto sortait à −0,8 / +5,6 /
    −16,0 / −12,6°, et les −16° du dos donnaient un rendu étiré dont le
    CONTOUR épousait la photo pendant que l'intérieur glissait. Une silhouette
    est un masque plein ; là où la forme reconstruite est fausse, elle répare
    une mauvaise géométrie avec une mauvaise caméra et son score MONTE.

    Et la silhouette ne mesure pas l'élévation à mieux que ±5° : balayée de
    −15 à +15, l'accord ne varie que de 0,01 à 0,03 par vue. D'où la
    préférence pour l'horizontale à accord égal.
    """
    foc_c, foc_f = _focale(rg.r_cal), _focale(rg.r_fin)
    memo = {}

    def score(e_deg):
        e_deg = round(float(e_deg), 3)
        if e_deg not in memo:
            el = math.radians(e_deg)
            memo[e_deg] = sum(_par_boite(V, F, alphas[r], foc_c, rg.r_cal,
                                         yaws[r], el)[0] for r in alphas)
        return memo[e_deg]

    # PAR VUE, et non sur la somme : réglée sur quatre vues, la même
    # pénalité clouait l'élévation d'une photo seule à 0°.
    def penalise(e_deg):
        return score(e_deg) / len(alphas) - rg.pente_niveau / 4 * abs(float(e_deg))

    grille = np.arange(-24., 24.1, 4.)
    if elev_depart is not None:
        grille = np.arange(elev_depart - 8., elev_depart + 8.1, 4.)
    e0 = max(grille, key=penalise)
    a, b = e0 - 4., e0 + 4.
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    while b - a > 0.4:
        if penalise(c) >= penalise(d):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
    e_deg = float(max((e0, c, d), key=penalise))
    elev = math.radians(e_deg)

    borne = math.radians(rg.roulis_libre if len(alphas) == 1 else rg.roulis_libre_mv)
    poses = {}
    for r in alphas:
        # Deux départs, sans roulis et avec celui des axes principaux : on
        # garde le meilleur, et la recherche par motifs le polit ensuite.
        ro = _roulis_de_depart(V, F, alphas[r], foc_c, rg.r_cal, yaws[r], elev, borne)
        departs = [(0.0,) + _par_boite(V, F, alphas[r], foc_c, rg.r_cal, yaws[r], elev)]
        if ro:
            departs.append((ro,) + _par_boite(V, F, alphas[r], foc_c, rg.r_cal,
                                              yaws[r], elev, ro))
        ro, _v, e, dx, dy = max(departs, key=lambda d: d[1])
        pas_ro = math.radians(2.0) if borne else 0.0
        _v, e, dx, dy, ro = _affiner(V, F, alphas[r], foc_c, rg.r_cal, yaws[r], elev,
                                     (e, dx, dy, ro), (0.03, 0.006, 0.006, pas_ro),
                                     (0.004, 0.0012, 0.0012, math.radians(0.25)))
        ro = max(-borne, min(borne, ro))
        v, e, dx, dy, ro = _affiner(V, F, alphas_fin[r], foc_f, rg.r_fin, yaws[r], elev,
                                    (e, dx, dy, ro),
                                    (0.004, 0.0012, 0.0012, pas_ro / 4),
                                    (0.0015, 0.0005, 0.0005, math.radians(0.1)))
        ro = max(-borne, min(borne, ro))
        poses[r] = {"echelle": e, "azimut": math.degrees(yaws[r]), "dx": dx,
                    "dy": dy, "roulis": math.degrees(ro), "accord": v}
    return poses, e_deg


# ------------------------------------------------------ garde-fous -----------
def _eroder(masque, n):
    m = torch.as_tensor(masque, dtype=torch.float32)[None, None]
    for _ in range(n):
        m = -torch.nn.functional.max_pool2d(-m, 3, stride=1, padding=1)
    return m[0, 0] > 0.5


def _falaises(prof, couvert, saut, marge):
    """Les pixels trop près d'un saut de profondeur, donc illisibles.

    Le test d'identité dit « ce texel est la surface la plus proche DANS LE
    RENDU », jamais « ce pixel, DANS LA PHOTO, appartient à cette pièce ».
    Là où la silhouette photographiée d'une pièce tombe deux pixels à côté de
    sa silhouette rendue, la pièce qui est DERRIÈRE reçoit ses pixels : sur la
    moto, le liseré chromé du silencieux était peint deux fois. Photos de
    1254 px sur un atlas de 4096 : un pixel photo vaut 3,3 texels.
    """
    p = prof.clone()
    p[~couvert] = p[couvert].max() + 10. * saut if bool(couvert.any()) else 0.
    g = torch.zeros_like(p)
    h = (p[:, 1:] - p[:, :-1]).abs()
    v = (p[1:, :] - p[:-1, :]).abs()
    g[:, :-1] = torch.maximum(g[:, :-1], h)
    g[:, 1:] = torch.maximum(g[:, 1:], h)
    g[:-1, :] = torch.maximum(g[:-1, :], v)
    g[1:, :] = torch.maximum(g[1:, :], v)
    f = (g > saut).float()[None, None]
    k = 2 * marge + 1
    f = torch.nn.functional.max_pool2d(f, (1, k), stride=1, padding=(0, marge))
    f = torch.nn.functional.max_pool2d(f, (k, 1), stride=1, padding=(marge, 0))
    return f[0, 0] > 0.5


# -------------------------------------------------- accord des couleurs ------
def _flou3d(vol, sigma):
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, device=vol.device, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    v = vol[:, None]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - dim)] = r
        pad[2 * (2 - dim) + 1] = r
        v = torch.nn.functional.conv3d(torch.nn.functional.pad(v, pad),
                                       k.reshape(shape))
    return v[:, 0]


def _harmoniser(P, base, photo, alpha, masque, appui, G=96,
                echelles=(1.5, 4.0, 10.0), teinte=1.0, teinte_max=None):
    """La texture dessinée, corrigée vers la couleur des photos voisines.

    On mesure le RAPPORT photo / dessinée là où les deux se recouvrent — un
    rapport et non une couleur, pour qu'une fuite entre pièces voisines ne
    repeigne rien — on le lisse DANS L'ESPACE 3D (dans l'atlas, deux texels
    voisins peuvent appartenir à deux pièces sans rapport), et on le prolonge
    là où aucune photo ne voit. La photo elle-même n'est jamais touchée.
    """
    dev = P.device
    sur = masque & (alpha > 0.9)
    if int(sur.sum()) < 1000:
        return base
    lr = (torch.log(photo.clamp(min=0.02)) - torch.log(base.clamp(min=0.02))
          ).clamp(-1.6, 1.6)
    lo = P[masque].min(0).values
    etendue = (P[masque].max(0).values - lo).max().clamp(min=1e-6)
    u = ((P - lo) / etendue).clamp(0, 1)
    t = (u * (G - 1)).round().long()
    idx = (t[:, 0] * G + t[:, 1]) * G + t[:, 2]
    num = torch.zeros(G ** 3, 3, device=dev).index_add_(0, idx[sur], lr[sur])
    den = torch.zeros(G ** 3, device=dev).index_add_(
        0, idx[sur], torch.ones(int(sur.sum()), device=dev))
    num = num.t().reshape(3, G, G, G)
    den = den.reshape(1, G, G, G)
    corr = torch.zeros(3, G, G, G, device=dev)
    pris = torch.zeros(1, G, G, G, device=dev)
    for sig in echelles:
        n, d = _flou3d(num, sig), _flou3d(den, sig)
        prend = (d / (d + appui)) * (1 - pris)
        corr = corr + (n / d.clamp(min=1e-6)) * prend
        pris = pris + prend
    corr = corr + lr[sur].median(0).values.reshape(3, 1, 1, 1) * (1 - pris)
    g = (u[:, [2, 1, 0]] * 2 - 1).reshape(1, -1, 1, 1, 3)
    c = torch.nn.functional.grid_sample(corr[None], g, mode="bilinear",
                                        padding_mode="border", align_corners=True)
    c = c[0, :, :, 0, 0].t()
    # LA LUMIÈRE EN ENTIER, LA TEINTE AVEC RETENUE. Le rapport moyen des
    # trois canaux dit « plus clair, plus sombre » ; ce qui s'en écarte dit
    # « plus vert, plus rouge », et c'est lui qui, prolongé loin des photos,
    # peignait des couleurs qu'aucune photo n'a.
    lum = c.mean(1, keepdim=True)
    ch = (c - lum) * teinte
    if teinte_max is not None:
        ch = ch.clamp(-teinte_max, teinte_max)
    return base * torch.exp(lum + ch)


# ------------------------------------------------------------ l'essentiel ----
def _preparer(photos):
    """Détourer les photos BRUTES comme le fait le pipeline : en lot quand il
    y en a plusieurs (même échelle pour toutes), sinon une à une. Rend
    {rôle: (image préparée, alpha dans le même cadre)}.

    NE SERT QUE HORS DU PIPELINE. Dans une génération, le pipeline a déjà
    détouré et passe `preparees` : le refaire coûterait un second chargement
    de BiRefNet, et donnerait un AUTRE alpha que celui qui a servi.
    """
    from . import matting
    m = matting.matter()
    try:
        if len(photos) > 1:
            lot = m.preparer_lot(photos)
        else:
            lot = {r: m.preparer_avec_geometrie(im) for r, im in photos.items()}
    finally:
        matting.liberer()
    return {r: (prep, geo.get("alpha")) for r, (prep, geo) in lot.items()}


def _azimuts(prepa):
    """Les angles mesurés par Depth Anything 3, ou None. Sur les sujets pauvres
    en relief il se tait plutôt que de se tromper ; on garde alors les angles
    ronds."""
    if len(prepa) != 4:
        return None
    try:
        from . import config
        from .multivue import cotes
        rapport = cotes.mesurer(config, {VERS_ROLE[r]: p for r, (p, _a)
                                         in prepa.items()})
        mesure = cotes.azimuts_mesures(rapport)
    except Exception as exc:                                    # noqa: BLE001
        log.info("projection : azimuts non mesurables (%s), angles ronds", exc)
        return None
    if not mesure:
        return None
    return {r: mesure[VERS_ROLE[r]] for r in ROLES if VERS_ROLE[r] in mesure}


def _texture_de(maillage):
    mat = maillage.visual.material
    tex = getattr(mat, "baseColorTexture", None)
    if tex is None:
        tex = getattr(mat, "image", None)
    return tex


def _camera_des_photos(V, F, UV, tex, prepa, azimuts, cotes_surs, rg: Reglages,
                       rapport: Rapport, jalon):
    """La caméra de chaque photo, sur le maillage `V` déjà centré et ramené à
    l'étendue unité : sens de la face, profils, direction d'une photo seule,
    élévation, taille, centrage et roulis. Rend (lacets, poses, élévation) et
    remplit `rapport`. C'est tout le calage : le banc l'appelle seul.
    """
    dev = V.device
    if azimuts is None:
        azimuts = _azimuts(prepa)
    rapport.azimuts_mesures = bool(azimuts)
    if cotes_surs is None:
        cotes_surs = bool(azimuts)
    yaws = {r: math.radians((azimuts or {}).get(r, ROLES[r])) for r in prepa}

    # --- le calage des caméras ---
    t_cal = time.time()
    al_cal, al_fin, ph_cal = {}, {}, {}
    for r, (pr, dec) in prepa.items():
        al_cal[r] = torch.as_tensor(np.asarray(
            dec.resize((rg.r_cal, rg.r_cal), Image.BILINEAR)) > 128, device=dev)
        al_fin[r] = torch.as_tensor(np.asarray(
            dec.resize((rg.r_fin, rg.r_fin), Image.BILINEAR)) > 128, device=dev)
        ph_cal[r] = torch.as_tensor(np.asarray(
            pr.convert("L").resize((rg.r_cal, rg.r_cal), Image.BILINEAR)),
            dtype=torch.float32, device=dev) / 255.
    # DE QUEL CÔTÉ EST LA FACE : mesuré, puis appliqué à toutes les vues.
    T_dessinee = torch.as_tensor(np.asarray(tex.convert("RGB")).copy(),
                                 dtype=torch.float32, device=dev) / 255.
    sens, echange, par = _sens_de_la_face(V, F, UV, T_dessinee, al_cal, ph_cal,
                                          yaws, rg, cotes_surs)
    del T_dessinee, ph_cal
    yaws = {r: y + sens + (math.pi if echange and r in ("droite", "gauche") else 0.0)
            for r, y in yaws.items()}
    rapport.sens_face, rapport.sens_decide_par = math.degrees(sens), par
    rapport.cotes_echanges = echange
    # UNE PHOTO SEULE : d'où a-t-elle été prise ?
    elev_depart = None
    if list(prepa) == ["face"] and not azimuts and rg.azimut_libre > 0:
        decalage, elev_depart = _azimut_de_la_photo(V, F, al_cal["face"],
                                                    yaws["face"], rg)
        yaws["face"] += decalage
        rapport.azimut_photo = math.degrees(decalage)
    jalon(0.10)
    poses, e_deg = _caler(V, F, al_cal, al_fin, yaws, rg, elev_depart)
    del al_cal, al_fin
    jalon(0.45)
    rapport.duree_calage = time.time() - t_cal
    rapport.elevation = e_deg
    rapport.vues = poses
    return yaws, poses, e_deg


def _fondu(P, Nt, masque, peint, rayon, cote, trous=3, pas=3, paquet=1_000_000):
    """Poids de 0 à 1 par texel peint : 0 au bord de la zone peinte, 1 à
    `rayon` (3D) et au-delà, en courbe douce. Les zones non peintes du côté
    opposé d'une paroi mince (normale contraire) ne comptent pas : l'autre face
    d'une lame n'est pas un bord.

    LE BORD EST CELUI DE LA ZONE, PAS CELUI DE SES TROUS. La zone peinte est
    criblée de trous d'un ou deux texels (un test de visibilité qui échoue) ;
    pris pour des bords, ils estompaient la photo presque partout (43 à 70 %
    de l'objet changeait sur le banc). On les bouche d'abord, comme le mélange
    franc (`trous_texels`)."""
    import numpy as np
    from scipy.spatial import cKDTree
    Fn = torch.nn.functional
    poids = torch.ones(peint.shape[0], dtype=torch.float32, device=peint.device)
    n = 2 * int(trous) + 1
    # hors du maillage (les gouttières entre îlots) compte comme peint : sinon un
    # texel non peint collé au bord d'un îlot n'est jamais bouché, et chacun
    # creusait une cellule sombre dans le fondu (motif en îlots sur la poupée)
    pe = (peint | ~masque).reshape(1, 1, cote, cote).float()
    ferme = (-Fn.max_pool2d(-Fn.max_pool2d(pe, n, 1, int(trous)), n, 1, int(trous))).reshape(-1) > 0.5
    del pe
    non = torch.nonzero(masque & ~peint & ~ferme).squeeze(1)[::pas]
    oui = torch.nonzero(peint).squeeze(1)
    if non.numel() == 0 or oui.numel() == 0:
        return poids
    arbre = cKDTree(P[non].cpu().numpy())
    nq = Nt[non].cpu().numpy()
    for i in range(0, oui.numel(), paquet):
        o = oui[i:i + paquet]
        dist, j = arbre.query(P[o].cpu().numpy(), k=8, distance_upper_bound=rayon, workers=-1)
        valide = np.isfinite(dist)
        jj = np.where(valide, j, 0)
        meme = valide & (np.einsum("nkc,nc->nk", nq[jj], Nt[o].cpu().numpy()) > 0.2)
        t = np.clip(np.where(meme, dist, rayon).min(1) / rayon, 0.0, 1.0)
        poids[o] = torch.as_tensor(t * t * (3 - 2 * t), dtype=torch.float32, device=poids.device)
    return poids


def _remplir_gouttieres(rgb, couvert, cote, n):
    """Chaque texel hors île reçoit, sur `n` texels, la moyenne de ses voisins
    déjà remplis : la couleur des îles déborde dans les gouttières, comme la
    cuisson le fait avant la projection, mais avec les couleurs d'après."""
    Fn = torch.nn.functional
    c = rgb.t().reshape(1, 3, cote, cote).contiguous()
    m = couvert.float().reshape(1, 1, cote, cote)
    for _ in range(n):
        num = Fn.avg_pool2d(c * m, 3, 1, 1)
        den = Fn.avg_pool2d(m, 3, 1, 1)
        neuf = (den > 0) & (m < 0.5)
        c = torch.where(neuf, num / den.clamp(min=1e-6), c)
        m = torch.maximum(m, (den > 0).float())
        del num, den, neuf
    return c.reshape(3, -1).t()


def _melange_franc(peint, meilleur, couleur, masque, cote, rg: Reglages):
    """Rend (peint, alpha) pour le mélange « franc ». Modifie `couleur` en place
    dans les petits trous bouchés."""
    Fn = torch.nn.functional
    grille = lambda t: t.reshape(1, 1, cote, cote)
    pe = grille(peint.float())
    k = int(rg.trous_texels)
    if k > 0:
        n = 2 * k + 1
        ferme = -Fn.max_pool2d(-Fn.max_pool2d(pe, n, 1, k), n, 1, k)
        trou = ((ferme > 0.5) & (pe < 0.5)).reshape(-1) & masque
        if bool(trou.any()):
            c = couleur.t().reshape(1, 3, cote, cote)
            num = Fn.avg_pool2d(c * pe, n, 1, k)
            den = Fn.avg_pool2d(pe, n, 1, k).clamp(min=1e-6)
            rempli = (num / den).reshape(3, -1).t()
            couleur[trou] = rempli[trou]
            meilleur[trou] = rg.franc_seuil
            peint = peint | trou
            del c, num, den, rempli
        del ferme, trou
    a_w = (meilleur / rg.franc_seuil).clamp(0, 1)
    # La distance au bord de la zone photographiée, en texels : l'hors-atlas
    # compte comme « photographié », sinon chaque bord d'île UV serait fondu.
    m = grille((peint | ~masque).float())
    d = torch.zeros_like(m)
    for _ in range(int(rg.bord_texels)):
        m = -Fn.max_pool2d(-m, 3, 1, 1)
        d = d + m
    rampe = (d / max(int(rg.bord_texels), 1)).clamp(0, 1).reshape(-1)
    return peint, a_w * rampe * peint.float()


def _recalage_par_relief(V, F, prepa, yaws, poses, elev, rapport: Rapport):
    """Un champ de déplacements par vue (voir `relief.py`), ou None."""
    from . import relief
    if not relief.disponible():
        log.info("projection : MoGe-2 absent, pas de recalage par le relief")
        return None
    rr = relief.ReglagesRelief()
    R = rr.cote
    foc = _focale(R)
    photos, masques, modeles = {}, {}, {}
    for r, (prep, al) in prepa.items():
        pz = poses[r]
        roul = math.radians(pz.get("roulis", 0.0))
        x, y, d = _proj(V, yaws[r], foc, R, elev, pz["echelle"], pz["dx"], pz["dy"], roul)
        (xyz,), cov = uvraster.camera(_clip(x, y, d, R), F, R, attributes=(V,))
        couvert = (cov.reshape(R, R) > 0).cpu().numpy()
        modeles[r] = (relief.normales_rendues(xyz.reshape(R, R, 3).cpu().numpy(), couvert,
                                              yaws[r], elev, roul), couvert)
        a = np.asarray(al.resize(prep.size, Image.BILINEAR) if al.size != prep.size
                       else al).astype(np.float32) / 255.
        rgb = np.asarray(prep.convert("RGB")).astype(np.float32) / 255.
        rgb = np.clip(rgb / np.clip(a, 1e-3, 1)[..., None], 0, 1)
        photo = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8))
                           .resize((R, R), Image.BICUBIC)).astype(np.float32) / 255.
        detour = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                            .resize((R, R), Image.BILINEAR)).astype(np.float32) / 255. > 0.5
        photos[r] = photo * detour[..., None]
        masques[r] = detour
        del xyz, cov
    normales = relief.normales_des_photos(photos)
    champs = {}
    for r in prepa:
        champs[r], part = relief.champ(normales[r], modeles[r][0], masques[r], modeles[r][1], rr)
        rapport.recalage_relief[r] = round(100 * part, 1)
    return champs


def projeter(maillage, photos=None, *, preparees=None, azimuts=None,
             cotes_surs=None, deformations=None, reglages: Reglages | None = None,
             jalon=None):
    """Recopie les vrais pixels des photos sur la texture de `maillage`.

    `maillage` : trimesh texturé (UV + baseColorTexture), TEL QU'IL SERA
    LIVRÉ — on ne change ni sommet ni UV, seulement l'image de texture. Son
    échelle et sa position n'importent pas ; son axe vertical doit être Y,
    celui du moteur avant `_orient_and_scale`.
    `preparees` : {rôle: (image préparée, alpha du même cadre)}, ce que le
    pipeline a déjà ; sinon `photos` : {rôle: PIL brute}, détourées ici.
    Rôles : "face" (obligatoire), "droite", "dos", "gauche".
    `azimuts` : {rôle: degrés} déjà mesurés. None : on les mesure ici s'il y
    a quatre vues. {} : on garde les angles ronds sans mesurer.
    `cotes_surs` : les profils « droite » et « gauche » sont-ils sûrs
    (mesurés, ou imposés par l'utilisateur) ? None : sûrs si les azimuts le
    sont. Sinon la projection vérifie elle-même qu'ils ne sont pas échangés.
    `jalon(fraction)` est appelé entre les étapes : c'est là qu'une
    annulation peut interrompre le travail.

    Rend un `Rapport`. Modifie la texture de `maillage` en place.
    """
    rg = reglages or Reglages()
    t0 = time.time()
    rapport = Rapport()
    tex = _texture_de(maillage)
    if tex is None or getattr(maillage.visual, "uv", None) is None:
        raise ValueError("maillage sans texture ni UV : rien à projeter")
    jalon = jalon or (lambda _f: None)
    if preparees is None:
        if not photos or photos.get("face") is None:
            raise ValueError("la photo de face est obligatoire")
        photos = {r: photos[r] for r in ROLES
                  if r in photos and photos[r] is not None}
        prepa = _preparer(photos)
    else:
        prepa = {r: preparees[r] for r in ROLES
                 if r in preparees and preparees[r] is not None}
    prepa = {r: v for r, v in prepa.items() if v[1] is not None}
    if "face" not in prepa:
        raise ValueError("la photo de face n'a pas de détourage exploitable")
    jalon(0.05)

    dev = "cuda"
    cote = tex.size[0]
    V = torch.as_tensor(np.asarray(maillage.vertices), dtype=torch.float32, device=dev)
    F = torch.as_tensor(np.asarray(maillage.faces), dtype=torch.int64, device=dev)
    UV = torch.as_tensor(np.asarray(maillage.visual.uv), dtype=torch.float32, device=dev)
    N = torch.as_tensor(np.asarray(maillage.vertex_normals).copy(),
                        dtype=torch.float32, device=dev)
    # LE REPÈRE DE LA CAMÉRA DU MOTEUR : le sujet centré sur sa boîte et
    # ramené à une étendue unité. Aucun retournement : mesuré sur la moto et
    # le samouraï, le maillage tel quel s'accorde nettement mieux aux photos
    # (0,89 contre 0,79), et un modèle généré n'est jamais sorti en miroir.
    V = V - (V.max(0).values + V.min(0).values) / 2
    V = V / (V.max(0).values - V.min(0).values).max()

    yaws, poses, e_deg = _camera_des_photos(V, F, UV, tex, prepa, azimuts,
                                            cotes_surs, rg, rapport, jalon)
    elev = math.radians(e_deg)
    if deformations is None and rg.recalage_relief:
        try:
            deformations = _recalage_par_relief(V, F, prepa, yaws, poses, elev, rapport)
        except Exception as exc:                              # noqa: BLE001
            # UN RECALAGE FIN QUI ÉCHOUE N'EST PAS UNE PROJECTION QUI ÉCHOUE :
            # la caméra rigide est là, on s'en contente et on le dit.
            log.warning("projection : recalage par le relief sauté (%s)", exc)
            deformations = None
        jalon(0.48)

    # --- l'atlas : la position 3D et la normale de chaque texel ---
    (pos, nrm), couvert = uvraster.atlas(UV, F, cote, [V, N], as_numpy=False)
    masque = couvert.reshape(-1)
    P = pos.reshape(-1, 3)
    Nt = torch.nn.functional.normalize(nrm.reshape(-1, 3), dim=1)
    del pos, nrm, couvert
    lin = (torch.arange(cote, device=dev, dtype=torch.float32) + 0.5) / cote
    uu, vv = torch.meshgrid(lin, lin, indexing="xy")
    UVt = torch.stack([uu.reshape(-1), 1.0 - vv.reshape(-1)], 1)
    del uu, vv, lin
    tol_uv = rg.tolerance_uv_texels / cote
    focal = _focale(cote)

    meilleur = torch.zeros(P.shape[0], device=dev)
    couleur = torch.zeros(P.shape[0], 3, device=dev)

    for i_vue, (role, (prep, al)) in enumerate(prepa.items()):
        jalon(0.50 + 0.40 * i_vue / len(prepa))
        larg, haut = prep.size
        a_full = np.asarray(al.resize((larg, haut), Image.BILINEAR)
                            if al.size != (larg, haut) else al)
        # LE DÉTOURAGE ACCEPTÉ. On n'érode plus que le CONTOUR EXTÉRIEUR : le
        # bord d'un trou intérieur (les motifs sombres d'une lame damassée,
        # pris pour du fond) n'a pas de halo à retirer, et l'érosion y jetait
        # la moitié de la lame. Au-dessus de 0,5 d'alpha, défaire la
        # prémultiplication reste propre.
        from scipy import ndimage
        plein = torch.as_tensor(ndimage.binary_fill_holes(a_full > 128))
        opaque = (_eroder(plein, rg.erosion) & torch.as_tensor(a_full > 128)).to(dev)
        rgb = np.asarray(prep.convert("RGB")).astype(np.float32) / 255.
        a01 = np.clip(a_full.astype(np.float32) / 255., 1e-3, 1.)
        # `matting` rend du PRÉMULTIPLIÉ SUR NOIR : on le défait.
        photo = _vers_lineaire(torch.as_tensor(np.clip(rgb / a01[..., None], 0, 1),
                                               device=dev))
        pz = poses[role]
        yaw, ech, dx, dy = yaws[role], pz["echelle"], pz["dx"], pz["dy"]
        roul = math.radians(pz.get("roulis", 0.0))

        # LE TAMPON D'IDENTITÉ : l'UV et la profondeur vues depuis la caméra,
        # à la résolution de l'atlas.
        x, y, d = _proj(V, yaw, focal, cote, elev, ech, dx, dy, roul)
        (uv_vu, prof_vu), cov = uvraster.camera(
            _clip(x, y, d, cote), F, cote, attributes=(UV, d[:, None]))
        bord = _falaises(prof_vu.reshape(cote, cote), cov.reshape(cote, cote),
                         rg.saut_falaise * ech, rg.marge_falaise).reshape(-1)
        uv_vu = uv_vu.reshape(-1, 2)
        cov_ecran = cov.reshape(-1)
        del prof_vu, cov

        tx, ty, _td = _proj(P, yaw, focal, cote, elev, ech, dx, dy, roul)
        dans = masque & (tx >= 0) & (tx < cote) & (ty >= 0) & (ty < cote)
        plat = ty.clamp(0, cote - 1).long() * cote + tx.clamp(0, cote - 1).long()
        lui_meme = (uv_vu[plat] - UVt).abs().amax(1) < tol_uv
        couvert_ecran = cov_ecran[plat] > 0
        franc = ~bord[plat]
        del uv_vu, cov_ecran, bord

        # L'axe de vue bascule avec l'appareil.
        axe = torch.tensor([math.sin(yaw) * math.cos(elev), math.sin(elev),
                            math.cos(yaw) * math.cos(elev)], device=dev)
        cos = (Nt * axe).sum(1)
        lu = torch.stack([tx / cote, ty / cote], 1).clamp(0, 1)
        # LE RECALAGE FIN, s'il y en a un pour cette vue : un champ de
        # déplacements, en pixels de sa propre grille, qui dit où lire la
        # photo pour que son relief tombe sur celui du modèle.
        if deformations and role in deformations:
            Dt = torch.as_tensor(np.asarray(deformations[role], dtype=np.float32),
                                 device=dev).permute(2, 0, 1)[None]
            g = (lu * 2 - 1).view(1, -1, 1, 2)
            dl = torch.nn.functional.grid_sample(Dt, g, mode="bilinear",
                                                 align_corners=False)[0, :, :, 0].t()
            lu = (lu + dl / torch.tensor([Dt.shape[-1], Dt.shape[-2]], device=dev)).clamp(0, 1)
            del Dt, g, dl
        px = (lu[:, 0] * larg).clamp(0, larg - 1).long()
        py = (lu[:, 1] * haut).clamp(0, haut - 1).long()
        sur_sujet = opaque[py, px]

        vu = dans & couvert_ecran & lui_meme & (cos > rg.cos_mini) & sur_sujet & franc
        w = ((cos - rg.cos_mini) / (1 - rg.cos_mini)).clamp(0, 1) * vu.float()
        # ON CHOISIT, ON NE MOYENNE PAS : chaque texel prend la photo qui le
        # voit le mieux, une et une seule.
        gagne = w > meilleur
        if bool(gagne.any()):
            couleur[gagne] = uvraster.sample(photo, lu)[gagne]
            meilleur[gagne] = w[gagne]
        pz["texels_vus"] = int(vu.sum())
        del tx, ty, dans, plat, lui_meme, couvert_ecran, franc, cos, lu, px, py
        del sur_sujet, vu, w, gagne, photo, opaque
        torch.cuda.empty_cache()

    jalon(0.92)
    # --- le mélange, en lumière linéaire ---
    peint = meilleur > 1e-6
    if rg.melange == "franc":
        peint, alpha = _melange_franc(peint, meilleur, couleur, masque, cote, rg)
    else:
        alpha = (meilleur.clamp(0, 1) ** 0.35) * peint.float()
    img0 = tex.convert("RGBA") if tex.mode in ("RGBA", "LA") else tex.convert("RGB")
    rgb0 = np.asarray(img0.convert("RGB")).astype(np.float32) / 255.
    base = _vers_lineaire(torch.as_tensor(rgb0, device=dev).reshape(-1, 3))
    if rg.harmonie:
        base = _harmoniser(P, base, couleur, alpha, masque, rg.appui_harmonie,
                           teinte=rg.harmonie_teinte, teinte_max=rg.harmonie_teinte_max)
    rayon_fondu = float(rg.fondu if len(prepa) == 1 else rg.fondu_multivue)
    if rayon_fondu > 0:
        # après l'harmonie, qui garde son appui plein sur la photo
        alpha = alpha * _fondu(P, Nt, masque, peint, rayon_fondu, cote, int(rg.trous_texels))
    out = _vers_srgb(base * (1 - alpha[:, None]) + couleur * alpha[:, None])
    if rg.gouttieres_texels > 0:
        out = _remplir_gouttieres(out, masque, cote, int(rg.gouttieres_texels))
    rapport.couverture = 100. * float(peint[masque].float().mean())
    img = Image.fromarray((out.reshape(cote, cote, 3).clamp(0, 1).cpu().numpy() * 255
                           + 0.5).astype(np.uint8))
    if img0.mode == "RGBA":
        img.putalpha(img0.getchannel("A"))
    mat = maillage.visual.material
    if hasattr(mat, "baseColorTexture"):
        mat.baseColorTexture = img
    else:
        mat.image = img

    del V, F, UV, N, P, Nt, UVt, meilleur, couleur, peint, alpha, base, out, masque
    torch.cuda.empty_cache()
    rapport.duree = time.time() - t0
    log.info("projection : %.1f %% de la surface en vraie photo, face %s "
             "(%s)%s, azimut de la photo %+.0f°, élévation %+.1f°, %d vue(s), "
             "%.0f s (dont calage %.0f s)",
             rapport.couverture, "devant" if rapport.sens_face == 0 else "derrière",
             rapport.sens_decide_par,
             ", profils échangés" if rapport.cotes_echanges else "",
             rapport.azimut_photo, rapport.elevation, len(prepa),
             rapport.duree, rapport.duree_calage)
    return rapport
