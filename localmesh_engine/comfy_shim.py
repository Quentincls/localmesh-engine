"""Minimal stand-ins for the two ComfyUI modules the vendored trellis2 imports.

The newer trellis2 we depend on lives inside a ComfyUI custom node, so it does
`import folder_paths` and `from comfy.utils import ProgressBar`. It uses exactly
two things: a models directory, and a progress bar it ticks during sampling.
Registering tiny modules under those names is far less invasive than patching
the vendored source, which we want to keep updatable.

The progress bar is not a stub: routing it to a callback is what lets the app
show real progress *inside* a sampling stage instead of a frozen percentage.
"""
from __future__ import annotations

import sys
import types
from typing import Callable, Optional

#: Set by the engine before a job so ticks reach the UI.
_sink: Optional[Callable[[int, int], None]] = None


def set_progress_sink(fn: Optional[Callable[[int, int], None]]) -> None:
    """Route sampler ticks to `fn(current, total)`. None disables."""
    global _sink
    _sink = fn



class ProgressBar:
    """Same surface as comfy.utils.ProgressBar, forwarding to our sink."""

    def __init__(self, total: int):
        self.total = max(int(total), 1)
        self.current = 0

    def update_absolute(self, value: int, total: Optional[int] = None,
                        preview: object = None) -> None:
        if total is not None:
            self.total = max(int(total), 1)
        self.current = int(value)
        if _sink is not None:
            try:
                _sink(self.current, self.total)
            except BaseException as exc:                       # noqa: BLE001
                # UNE ANNULATION N'EST PAS UN INCIDENT DE BARRE.
                #
                # Ce `except` existe pour une bonne raison : un rapport de
                # progression ne doit jamais casser une generation. Mais il
                # attrapait AUSSI le `JobCancelled` que le puits leve quand
                # l'artiste annule — leve, puis jete. Pendant tout
                # l'echantillonnage de la forme, quatre-vingt-cinq secondes,
                # le bouton Annuler ne faisait donc rien du tout : le seul
                # point de controle de cette etape etait desamorce ici.
                #
                # `remaillage._tic` porte deja ce constat et cette correction
                # pour les jalons de texturation. Le meme piege vivait ici,
                # sur l'etape la plus longue a etre annulable.
                #
                # Reconnu par son NOM et non par son type : l'importer
                # creerait un cycle, ce module etant importe par `pipeline`.
                if type(exc).__name__ == "JobCancelled":
                    raise

    def update(self, value: int = 1) -> None:
        self.update_absolute(self.current + int(value))


def install(models_dir) -> None:
    """Register the shim modules. Idempotent; never shadows a real ComfyUI."""
    if "folder_paths" in sys.modules and not getattr(
            sys.modules["folder_paths"], "_lumengen_shim", False):
        return  # a genuine ComfyUI is present, leave it alone

    fp = types.ModuleType("folder_paths")
    fp.models_dir = str(models_dir)
    fp.base_path = str(models_dir)
    fp.get_folder_paths = lambda *_a, **_k: [str(models_dir)]
    fp.get_full_path = lambda _folder, name: str(models_dir) + "/" + str(name)
    fp._lumengen_shim = True

    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = ProgressBar
    utils._lumengen_shim = True

    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    comfy.__path__ = getattr(comfy, "__path__", [])
    comfy.utils = utils
    comfy._lumengen_shim = True

    sys.modules["folder_paths"] = fp
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
