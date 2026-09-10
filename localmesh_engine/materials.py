"""Material finishing: transparency mode, and the maps that ship with the asset.

TRELLIS.2 emits an opacity channel alongside base colour, roughness and
metalness, but the exported glTF material is written with `alphaMode: OPAQUE`.
Every viewer therefore ignores the alpha, and the capability is invisible -
upstream's own note says transparency "requires manual activation in 3D
software". Nobody would ever find it.

Rather than asking the artist to pick a glTF alpha mode, the mode is read off
the alpha the model actually produced. It is a measurement, not a preference:
an object is made of glass or it is not.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

log = logging.getLogger("localmesh_engine.materials")

AlphaMode = Literal["OPAQUE", "MASK", "BLEND"]

#: EN DESSOUS DE QUOI L OBJET EST PLEIN, MEME S IL A DES VITRES.
#:
#: C etait un vingtieme de la surface, et c etait trop bas. Un abribus genere
#: le 27 aout 2026 est sorti a 9,06 % — ses vitres, qui sont bien
#: translucides — et le materiau ENTIER est passe en BLEND. Resultat : la
#: tole, les montants et le mat devenaient fantomes avec elles. Retour de
#: Quentin : « j ai des maillages qui sont transparents parfois ».
#:
#: Il n y a QU UN materiau pour tout l objet. Declarer ce materiau
#: translucide, c est le declarer pour la tole autant que pour le verre.
#:
#: MESURE SUR LES QUATORZE MAILLAGES DE LA BIBLIOTHEQUE : douze objets pleins
#: entre 0,00 et 0,04 %, l abribus a 9,06 %, et rien entre les deux. Aucun
#: objet reellement en verre dans l echantillon. La barre est donc posee dans
#: le desert qui separe « un objet plein qui a des vitres » de « un objet en
#: verre », dont l atlas serait translucide de bout en bout.
#:
#: ET LE PARI EST ASYMETRIQUE, ce qui justifie de viser large. BLEND fait
#: perdre l ordre d affichage sur TOUT l objet — le prix de l erreur est
#: global. Quelques texels qui auraient du laisser passer la lumiere et qui
#: ne le font pas, ca ne se voit pas. Dans le doute : plein.
_OPAQUE_CEILING = 0.25
#: Alpha within this distance of 0 or 255 counts as "hard" - a cutout rather
#: than a gradient.
_HARD_MARGIN = 24
#: EN DESSOUS DE QUOI ON VOIT VRAIMENT A TRAVERS.
#:
#: La detection comptait comme "non opaque" tout texel sous 231, soit a plus
#: d'un pas de _HARD_MARGIN du blanc. Ca marchait sur l'alpha de l'ancienne
#: chaine, une decoupe nette heritee du splat. L'atlas cuit par TRELLIS a une
#: longue traine douce : mesure sur un personnage, 20,5 % de la SURFACE entre
#: 24 et 231, pour seulement 2,6 % reellement sous 128. La detection lisait
#: cet anticrenelage comme de la matiere translucide et sortait BLEND — un
#: objet plein rendu fantome, troue de partout.
#:
#: 128 est le point ou la transparence se voit : moitie du fond derriere.
#: Au-dessus, c'est du bruit de cuisson. Un vrai verre met bien plus d'un
#: vingtieme de sa surface sous cette barre ; un bonhomme de bois, non.
_VISIBLE_CEILING = 128


@dataclass
class MaterialReport:
    mode: AlphaMode
    detected: AlphaMode
    forced: bool
    non_opaque_ratio: float
    hard_ratio: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "detected": self.detected,
            "forced": self.forced,
            "non_opaque_pct": round(self.non_opaque_ratio * 100, 2),
            "reason": self.reason,
        }


def detect_alpha_mode(alpha: np.ndarray) -> tuple[AlphaMode, float, float, str]:
    """Classify an alpha channel into a glTF alpha mode.

    OPAQUE - practically everything is solid.
    MASK   - two populations, fully on or fully off: foliage, grilles, lace.
    BLEND  - a spread of intermediate values: glass, liquid, gauze.
    """
    total = alpha.size
    if total == 0:
        return "OPAQUE", 0.0, 0.0, "aucune donnée alpha"

    non_opaque = alpha < _VISIBLE_CEILING
    non_opaque_ratio = float(non_opaque.mean())

    if non_opaque_ratio < _OPAQUE_CEILING:
        return ("OPAQUE", non_opaque_ratio, 0.0,
                f"{non_opaque_ratio:.2%} de la surface non opaque — objet plein")

    # Among the non-solid texels, are they near-zero (a cutout) or in between?
    values = alpha[non_opaque]
    hard = float((values < _HARD_MARGIN).mean())
    if hard > 0.7:
        return ("MASK", non_opaque_ratio, hard,
                f"{non_opaque_ratio:.1%} de découpe franche — ajours ou feuillage")
    return ("BLEND", non_opaque_ratio, hard,
            f"{non_opaque_ratio:.1%} de valeurs intermédiaires — matière translucide")


#: Part des aretes de bord au-dela de laquelle on considere que la surface est
#: VRAIMENT ouverte. Quelques aretes isolees sont du bruit de maillage : les
#: compter ferait basculer chaque objet en double face pour rien.
_BORD_OUVERT_MINI = 0.001


def _bords_ouverts(geom) -> bool:
    """La surface livree est-elle ouverte au point qu il faille voir son dos ?

    POURQUOI CETTE MESURE EXISTE. Le code posait `doubleSided = False` partout,
    avec ce motif : un objet genere par TRELLIS est ferme, une face arriere ne
    sert donc a rien et coute cher. Le banc du 7 septembre 2026 a mesure le
    contraire sur les objets livres : 1 339 aretes de bord sur le samourai en
    quatre vues, 2 387 sur une graine du feu. Sur ces objets le dos disparait
    chez le client — et pas chez nous, parce que notre visionneuse force le
    double face de son cote.

    IL FAUT SOUDER AVANT DE COMPTER. Le depliage UV duplique les sommets le
    long de chaque couture : sur les indices bruts, 39 % des aretes du feu
    paraissent ouvertes alors qu il n en a qu UNE. On regroupe donc les
    sommets par position avant de chercher les aretes qui n appartiennent
    qu a une seule face. Meme convention que le banc, au micron.
    """
    try:
        v = np.asarray(getattr(geom, "vertices", None), dtype=np.float64)
        f = np.asarray(getattr(geom, "faces", None))
        if v.ndim != 2 or f.ndim != 2 or not len(f):
            return False
        _, inverse = np.unique(np.round(v, 6), axis=0, return_inverse=True)
        soude = inverse[f]
        aretes = np.sort(np.stack([soude[:, [0, 1]], soude[:, [1, 2]],
                                   soude[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        _, comptes = np.unique(aretes, axis=0, return_counts=True)
        bords = int((comptes == 1).sum())
        total = int(len(comptes))
        ouvert = total > 0 and bords > total * _BORD_OUVERT_MINI
        log.info("materiau : %d aretes de bord sur %d (%.3f %%), %s",
                 bords, total, 100.0 * bords / max(total, 1),
                 "double face" if ouvert else "simple face")
        return ouvert
    except Exception as exc:                                     # noqa: BLE001
        # Une surface qu on ne sait pas mesurer garde l ancien comportement.
        log.warning("aretes de bord non mesurees (%s) : simple face", exc)
        return False


def apply_alpha_mode(glb, requested: Optional[str] = None) -> Optional[MaterialReport]:
    """Read the alpha the model produced, and write the matching glTF mode.

    `requested` overrides the detection when the artist disagrees; "auto" or
    None keeps the measurement.
    """
    geoms = list(glb.geometry.values()) if hasattr(glb, "geometry") else [glb]
    report: Optional[MaterialReport] = None

    for geom in geoms:
        mat = getattr(getattr(geom, "visual", None), "material", None)
        if mat is None:
            continue
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None or tex.mode not in ("RGBA", "LA"):
            continue

        alpha = np.asarray(tex)[..., -1]
        detected, non_opaque, hard, reason = detect_alpha_mode(alpha)

        forced = bool(requested) and requested.lower() != "auto"
        mode: AlphaMode = requested.upper() if forced else detected  # type: ignore[assignment]

        mat.alphaMode = mode
        # `o_voxel.to_glb` ecrit doubleSided des qu'il ne remaille pas, et
        # LumenGen ne remaille jamais a ce stade : tous les objets sortaient
        # a double face. Sur un solide c'est faux et couteux — et couple a un
        # alphaMode trop permissif, ca donne exactement le rendu qu'on a vu :
        # on voit l'interieur au travers de sa propre surface.
        #
        # 3.3 : ON MESURE AU LIEU DE SUPPOSER. La ligne disait « un objet
        # genere par TRELLIS est ferme » ; le banc l'a dementi (voir
        # `_bords_ouverts`). Un solide garde le simple face, une surface
        # reellement ouverte garde son dos visible.
        mat.doubleSided = _bords_ouverts(geom)

        if mode == "MASK":
            # Halfway between the two populations; anything softer leaves
            # halos, anything harder eats thin structures.
            mat.alphaCutoff = 0.5
        elif mode == "OPAQUE":
            mat.alphaCutoff = None
            # UN OBJET OPAQUE NE PART PAS AVEC UN CANAL ALPHA.
            #
            # Dire « opaque » dans le materiau ne suffisait pas : l'atlas
            # continuait de porter son alpha, et l'alpha que TRELLIS.2
            # fabrique RECOPIE LA LUMINANCE -- il tombe partout ou la texture
            # est sombre. Sur une cassette noire, 53 % de l'atlas se retrouve
            # ainsi marque translucide alors que la couleur, elle, est juste.
            #
            # CE QUE CA DONNE, ET POURQUOI C'ETAIT INVISIBLE ICI. La norme
            # glTF dit d'ignorer l'alpha en mode OPAQUE, donc une visionneuse
            # exacte n'y voit rien. Toutes ne le sont pas : celle qui la lit
            # rend une cassette EN VERRE, on voit l'interieur au travers de sa
            # propre surface, et le client croit son objet rate. Le 9 septembre
            # 2026 c'est exactement ce qui est sorti du banc.
            #
            # On ne se fie donc pas a la bonne volonte du lecteur : le canal
            # part. L'atlas y perd un quart de son poids au passage.
            if getattr(tex, "mode", None) in ("RGBA", "LA"):
                mat.baseColorTexture = tex.convert(
                    "RGB" if tex.mode == "RGBA" else "L")

        report = MaterialReport(
            mode=mode, detected=detected, forced=forced,
            non_opaque_ratio=non_opaque, hard_ratio=hard, reason=reason,
        )
        log.info("material: %s (detected %s) - %s", mode, detected, reason)

    return report
