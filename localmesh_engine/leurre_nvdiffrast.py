"""Un leurre nommé `nvdiffrast`, pour que la wheel d'o_voxel s'importe sans lui.

## Pourquoi

`nvdiffrast` est sous NVIDIA Source Code License : **usage non commercial**.
LocalMesh est vendu. Il n'a donc rien à faire ni dans le paquet livré, ni sur
le disque d'un client.

Mais la wheel `o_voxel`, elle, l'importe au niveau module :

    o_voxel/__init__.py   ->  from . import postprocess
    o_voxel/postprocess.py:10  import nvdiffrast.torch as dr

Retirer nvdiffrast du disque rend donc `import o_voxel` impossible, et avec
lui toute génération. Le correctif ne peut pas être « on remplace la fonction
après coup » : il faut que l'import passe.

## Ce que ce module fait, et ne fait pas

Il inscrit dans `sys.modules` deux noms — `nvdiffrast` et `nvdiffrast.torch` —
qui contiennent exactement les deux symboles que `postprocess.py` nomme, et
rien d'autre. Aucun code NVIDIA, aucun algorithme repris : deux objets qui
LÈVENT dès qu'on essaie de s'en servir, en disant pourquoi.

Ils ne sont jamais appelés, parce que `patches.py` remplace
`o_voxel.postprocess.to_glb` par la version de `_vendu_o_voxel`, qui rasterise
avec `uvraster`. Le leurre n'existe que pour la seconde qui sépare l'import de
ce remplacement.

Si un jour quelque chose d'autre dans o_voxel appelle vraiment le rastériseur,
ça ne passera pas en silence : l'exception nomme la licence et le remplaçant.

## Ordre d'installation, et il compte

`installer()` doit être appelé AVANT le premier `import o_voxel`. Un vrai
nvdiffrast déjà présent sur la machine a la priorité : on n'écrase pas ce qui
est là — le leurre ne sert qu'aux installations propres, celles qu'on livre.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types

log = logging.getLogger("localmesh_engine.leurre_nvdiffrast")

_RAISON = (
    "nvdiffrast a été retiré de LocalMesh : sa licence (NVIDIA Source Code "
    "License) interdit l'usage commercial. La rastérisation de l'atlas UV "
    "passe par localmesh_engine.uvraster. Si vous lisez ceci, un chemin d'o_voxel "
    "appelle encore le rastériseur d'origine et il faut le porter aussi."
)


class _Refus:
    """Tout ce qu'on en fait lève, en disant quoi faire à la place."""

    def __init__(self, quoi: str) -> None:
        self._quoi = quoi

    def __call__(self, *_a, **_k):
        raise RuntimeError(f"{self._quoi} : {_RAISON}")


def _sous_le_runtime(chemin: str | None) -> bool:
    """Ce fichier vit-il sous la racine du runtime LocalMesh ?"""
    racine = os.environ.get("LUMENGEN_ROOT")
    if not racine:
        # Lancé sans la variable (un banc, un script) : la racine que
        # config a résolue vaut autant.
        try:
            from . import config as _config
            racine = str(_config.RUNTIME_ROOT)
        except Exception:                                     # noqa: BLE001
            racine = None
    if not racine or not chemin:
        return False
    try:
        return os.path.normcase(os.path.abspath(chemin)).startswith(
            os.path.normcase(os.path.abspath(racine)) + os.sep)
    except Exception:                                     # noqa: BLE001
        return False


def _deja_present() -> bool:
    """Un vrai nvdiffrast est-il installé ? On ne l'écrase pas.

    SAUF S'IL EST À NOUS. Les installations d'avant runtime-2.zip ont
    reçu nvdiffrast dans le runtime LocalMesh, et il était importé à
    chaque génération sans jamais servir (usage non commercial). Sous
    notre racine, le leurre prime ; ailleurs (le banc, test_uvraster),
    le vrai module garde la main (audit du 3 septembre 2026).
    """
    if "nvdiffrast" in sys.modules:
        return True
    try:
        spec = importlib.util.find_spec("nvdiffrast")
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origine = spec.origin or next(iter(spec.submodule_search_locations or []), None)
    if _sous_le_runtime(origine):
        log.info("nvdiffrast du runtime ignoré, le leurre prime : %s", origine)
        return False
    return True


def verifier() -> None:
    """Au journal si le VRAI nvdiffrast est monté : ce ne doit jamais
    arriver sur une installation cliente."""
    m = sys.modules.get("nvdiffrast")
    if m is not None and not getattr(m, "_leurre", False):
        log.warning("le vrai nvdiffrast est chargé : %s",
                    getattr(m, "__file__", "?"))


def installer() -> bool:
    """Poser le leurre. Rend True s'il a été pose, False s'il n'a rien à faire."""
    if _deja_present():
        return False

    torche = types.ModuleType("nvdiffrast.torch")
    torche.RasterizeCudaContext = _Refus("nvdiffrast.torch.RasterizeCudaContext")
    torche.RasterizeGLContext = _Refus("nvdiffrast.torch.RasterizeGLContext")
    torche.rasterize = _Refus("nvdiffrast.torch.rasterize")
    torche.interpolate = _Refus("nvdiffrast.torch.interpolate")
    torche.texture = _Refus("nvdiffrast.torch.texture")
    torche.antialias = _Refus("nvdiffrast.torch.antialias")

    racine = types.ModuleType("nvdiffrast")
    racine.__path__ = []          # un paquet, pour que `nvdiffrast.torch` resolve
    racine._leurre = True
    racine.torch = torche

    sys.modules["nvdiffrast"] = racine
    sys.modules["nvdiffrast.torch"] = torche
    log.info("leurre nvdiffrast pose : la rasterisation passe par uvraster")
    return True
