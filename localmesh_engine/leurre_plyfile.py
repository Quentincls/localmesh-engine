"""Un leurre nommé `plyfile`, pour que la wheel d'o_voxel s'importe sans lui.

## Pourquoi

`plyfile` est sous GPL-3.0 : il n'a rien à faire dans un produit vendu fermé.
Mais la wheel `o_voxel` l'importe au niveau module :

    o_voxel/__init__.py       ->  from . import io
    o_voxel/io/__init__.py:3      from .ply import *
    o_voxel/io/ply.py:5           import plyfile

Retirer plyfile du disque rend donc `import o_voxel` impossible, et avec lui
toute génération. Il faut que l'import passe.

## Ce que ce module fait, et ne fait pas

Il inscrit dans `sys.modules` un module `plyfile` qui porte les noms que le
code vendu nomme, et rien d'autre. Aucune ligne de plyfile n'est reprise : des
objets qui LÈVENT dès qu'on s'en sert, en disant pourquoi.

Aucun chemin du moteur ne lit ni n'écrit de PLY par plyfile. Les seuls qui le
feraient : `o_voxel.io.read/write` sur un `.ply` (appelés par `Voxel.save` et
`Voxel.load`, que le moteur n'appelle pas) et `utils3d.io.read_ply/write_ply`
(import dans la fonction, jamais appelées). Si l'un d'eux sert un jour, ça ne
passera pas en silence : l'exception nomme la licence et dit quoi faire.

## Ordre d'installation, et il compte

`installer()` AVANT le premier `import o_voxel`, juste après le leurre
nvdiffrast, dans `config.bootstrap`. Même règle que lui : un vrai plyfile HORS
de la racine du runtime garde la main (banc, développeur) ; SOUS la racine, le
leurre prime, pour que les installations qui l'ont encore sur le disque
cessent de le charger.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import types

from .leurre_nvdiffrast import _sous_le_runtime

log = logging.getLogger("localmesh_engine.leurre_plyfile")

_RAISON = (
    "LocalMesh does not use plyfile for PLY processing. Replace calls to "
    "o_voxel.io.read/write('.ply') or utils3d.io.*_ply with trimesh or NumPy."
)

#: Les paquets qui ne doivent JAMAIS être chargés pendant une génération,
#: même s'ils traînent sur le disque d'une ancienne installation : licence non
#: commerciale (meshlib, nvdiffrec_render), provenance Tencent sans licence
#: (custom_rasterizer), ffmpeg compilé en GPL (imageio_ffmpeg). Rien du moteur
#: ne les importe ; `verifier()` le dit au journal si ça change.
AUTRES_INTERDITS = ("meshlib", "mrmeshpy", "nvdiffrec_render",
                    "custom_rasterizer", "custom_rasterizer_kernel",
                    "imageio_ffmpeg")


class _Refus:
    """Tout ce qu'on en fait lève : appel comme attribut."""

    def __init__(self, quoi: str) -> None:
        object.__setattr__(self, "_quoi", quoi)

    def __call__(self, *_a, **_k):
        raise RuntimeError(f"{self._quoi}: {_RAISON}")

    def __getattr__(self, nom: str):
        if nom.startswith("__"):
            raise AttributeError(nom)
        raise RuntimeError(f"{self._quoi}.{nom}: {_RAISON}")

    def __repr__(self) -> str:
        return f"<stub for {self._quoi}>"


class PlyParseError(Exception):
    """Nommée pour qu'un `except plyfile.PlyParseError` reste valide."""


class PlyElementParseError(PlyParseError):
    pass


class PlyHeaderParseError(PlyParseError):
    pass


def _deja_present() -> bool:
    """Un vrai plyfile est-il installé hors de notre racine ? On ne l'écrase pas."""
    if "plyfile" in sys.modules:
        return True
    try:
        spec = importlib.util.find_spec("plyfile")
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    if _sous_le_runtime(spec.origin):
        log.info("Using the plyfile stub instead of the runtime copy: %s", spec.origin)
        return False
    return True


def verifier() -> None:
    """Au journal si le VRAI plyfile, ou un autre paquet interdit, est chargé :
    ce ne doit jamais arriver sur une installation cliente."""
    m = sys.modules.get("plyfile")
    if m is not None and not getattr(m, "_leurre", False):
        log.warning("The real plyfile package (GPL-3.0) is loaded: %s",
                    getattr(m, "__file__", "?"))
    charges = sorted(n for n in list(sys.modules)
                     if n.split(".")[0] in AUTRES_INTERDITS)
    if charges:
        log.warning("Restricted package loaded: %s", ", ".join(charges))


def installer() -> bool:
    """Poser le leurre. Rend True s'il a été posé, False s'il n'a rien à faire."""
    if _deja_present():
        return False
    mod = types.ModuleType("plyfile")
    # Une spec, pour qu'un `importlib.util.find_spec("plyfile")` venu d'une
    # autre bibliothèque réponde au lieu de lever ValueError.
    mod.__spec__ = importlib.machinery.ModuleSpec("plyfile", None)
    mod._leurre = True
    for nom in ("PlyData", "PlyElement", "PlyProperty", "PlyListProperty"):
        setattr(mod, nom, _Refus(f"plyfile.{nom}"))
    mod.PlyParseError = PlyParseError
    mod.PlyElementParseError = PlyElementParseError
    mod.PlyHeaderParseError = PlyHeaderParseError
    sys.modules["plyfile"] = mod
    log.info("Installed the plyfile stub")
    return True
