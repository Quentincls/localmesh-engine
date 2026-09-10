# -*- coding: utf-8 -*-
"""La séquence complète, d'quatre photos détourées au maillage brut.

C'est le seul point d'entrée du multivue. Tout ce qui vient avant, le
détourage, et tout ce qui vient après, le nettoyage, la texture et la
finition, appartient à la voie normale : le multivue ne change QUE la façon
d'obtenir la forme.

Ce choix n'est pas une économie de code, c'est ce qui rend les deux voies
comparables. Une différence à l'écran vient alors de la forme, et de rien
d'autre.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

from . import MultivueIndisponible, Poids, parite as _parite
from . import forme as _forme
from . import structure as _structure
from . import traits as _traits

_RECETTE = None


def recette() -> dict:
    """La recette gelée, lue une fois.

    Elle vit dans un fichier à côté du code plutôt que dans des constantes :
    un réglage qu'on peut lire sans ouvrir un module est un réglage qu'on
    n'oublie pas de mettre à jour.
    """
    global _RECETTE
    if _RECETTE is None:
        _RECETTE = json.loads(
            (Path(__file__).parent / "recette.json").read_text(encoding="utf-8"))
    return _RECETTE


def generer(config, images: dict, graine: int, *, lateralite: str = "auto",
            decision_mesuree: str | None = None, geometrie: dict | None = None,
            azimuts: dict | None = None, accord: dict | None = None,
            pas_structure: int | None = None, pas_forme: int | None = None,
            force_imposee: float | None = None,
            jalon=None):
    """Quatre vues détourées, un maillage brut.

    `images` porte, par rôle (`front`, `right`, `back`, `left`), un tableau
    HxWx3 de flottants entre 0 et 1.

    `jalon(nom, part)` reçoit l'avancement, comme dans la voie normale.
    """
    def dire(nom, part):
        if jalon:
            jalon(nom, part)

    poids = Poids.depuis(config.MODELS_ROOT)
    manquants = poids.manquants()
    if manquants:
        raise MultivueIndisponible(manquants)

    cotes = _parite(lateralite, decision_mesuree)
    # LA FORCE SUIT LE DESACCORD DES PHOTOS : voir `structure.SEUIL_DESACCORD`
    # pour la mesure qui l'impose. `force_imposee` est la manette du banc :
    # sans elle, comparer les deux regimes sur un meme sujet demandait de
    # redemarrer le moteur avec une variable d'environnement.
    force = (_structure.force_selon_desaccord(accord)
             if force_imposee is None else float(force_imposee))
    r = recette()

    dire("encoding views", 0.10)
    traits_512 = _traits.encoder(config, images, poids, taille=512)

    dire("merging views", 0.25)
    cellules = _structure.volume(
        config, _traits.jetons_seuls(traits_512), poids, graine, cotes,
        r, geometrie, azimuts, force, pas_structure)

    # L'ENCODAGE FIN VIENT APRÈS LA STRUCTURE, ET C'EST VOULU. Les traits en
    # 1024 pèsent quatre fois ceux en 512 ; les calculer d'avance les ferait
    # attendre en mémoire pendant toute l'étape de fusion, pour rien.
    dire("encoding views", 0.35)
    traits_1024 = _traits.encoder(config, images, poids, taille=1024)

    dire("generating shape", 0.45)
    maillage = _forme.maillage(
        config, {512: traits_512, 1024: traits_1024}, cellules, poids,
        graine, cotes, r, geometrie, azimuts, force, pas_forme)

    del traits_512, traits_1024
    gc.collect()
    return maillage, {"cellules": int(len(cellules)), "lateralite": cotes,
                      "azimuts": azimuts, "force": force}
