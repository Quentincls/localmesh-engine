"""Le vrai pic de mémoire vidéo d'un travail, y compris hors PyTorch.

`torch.cuda.max_memory_allocated()` ne compte que ce que PyTorch a demandé.
C'est exactement ce qui manquait : pendant une conversion, cuBVH (occlusion)
et cumesh (dépliage UV) allouent en CUDA brut, de leur côté, et leur coût
n'apparaissait nulle part. Un testeur en 8 Go qui déborde nous envoyait donc
un manifeste annonçant « 0 Go ».

**NVML ne sait pas répondre par processus sous Windows.** En mode WDDM —
celui de toutes les GeForce — `nvmlDeviceGetComputeRunningProcesses` liste
bien les processus mais rend `usedGpuMemory = None` pour chacun. Vérifié sur
la machine de développement, sur les trois variantes de l'appel. Comme
LocalMesh est Windows-only, cette porte est fermée pour de bon : il ne faut
pas construire dessus.

On mesure donc deux choses, et on les rapporte séparément parce qu'elles ne
répondent pas à la même question :

* **`pic_gb`, ce que le travail a pris** — le maximum de la mémoire occupée
  sur la carte MOINS ce qui l'était déjà au démarrage du travail. Approché,
  puisqu'un autre programme peut bouger pendant ce temps ; jamais inférieur
  au pic réservé par PyTorch, qui est à nous avec certitude.
* **`occupation_pct`, à quel point la carte était pleine** — la vraie cause
  des ralentissements chez les testeurs. Au-delà de ~95 %, Windows bascule
  en mémoire partagée et une étape de quatre secondes en prend des minutes
  (déjà diagnostiqué en 0.9.5). C'est ce chiffre-là qui explique un
  « c'est hyper lent », pas le nôtre pris isolément.

Un pic ne se lit pas après coup : on échantillonne dans un fil pendant tout
le travail.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("localmesh_engine.vrammetre")

#: Assez fin pour attraper le pic d'une allocation qui dure une seconde,
#: assez lâche pour ne rien coûter (NVML se lit en ~0,3 ms).
_PERIODE = 0.25


def _poignee():
    """La poignée NVML du GPU 0, ou None. Initialise NVML au besoin."""
    try:
        import pynvml

        try:
            pynvml.nvmlDeviceGetCount()
        except Exception:                                        # noqa: BLE001
            pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:                                            # noqa: BLE001
        return None


def _occupation(poignee) -> tuple[float, float]:
    """(occupé, total) en Go sur la carte, tous programmes confondus."""
    import pynvml

    mem = pynvml.nvmlDeviceGetMemoryInfo(poignee)
    return mem.used / 1024 ** 3, mem.total / 1024 ** 3


class Metre:
    """Échantillonne l'occupation de la carte tant qu'il tourne.

    S'utilise en gestionnaire de contexte ; `pic_gb`, `occupation_pct` et
    `portee` sont lisibles après la sortie. Ne lève jamais : une mesure ratée
    ne doit pas coûter une conversion.
    """

    def __init__(self) -> None:
        self.pic_gb = 0.0
        self.occupation_pct = 0.0
        self.carte_gb = 0.0
        self.portee = "indisponible"
        self._base_gb = 0.0
        self._pic_carte_gb = 0.0
        self._stop = threading.Event()
        self._fil: threading.Thread | None = None

    def _boucle(self, poignee) -> None:
        while True:
            try:
                occupe, _total = _occupation(poignee)
                self._pic_carte_gb = max(self._pic_carte_gb, occupe)
            except Exception:                                    # noqa: BLE001
                pass
            if self._stop.is_set():
                return                       # une dernière lecture, puis fin
            self._stop.wait(_PERIODE)

    def __enter__(self) -> "Metre":
        # Le compteur de PyTorch est remis à zéro pour que son pic parle de CE
        # travail et non du maximum jamais atteint depuis le démarrage — un
        # chiffre qui ne redescend jamais et n'apprend donc rien.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:                                        # noqa: BLE001
            pass
        poignee = _poignee()
        if poignee is None:
            return self
        try:
            self._base_gb, self.carte_gb = _occupation(poignee)
            self._pic_carte_gb = self._base_gb
        except Exception:                                        # noqa: BLE001
            return self
        self._fil = threading.Thread(target=self._boucle, args=(poignee,),
                                     name="vrammetre", daemon=True)
        self._fil.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._fil is not None:
            self._fil.join(timeout=2.0)
        torche = _pic_torch()
        if self.carte_gb > 0.0:
            # Le delta peut sous-estimer si un autre programme a libéré de la
            # place pendant le travail ; le pic réservé par PyTorch, lui, est
            # à nous sans discussion. On garde le plus grand des deux.
            self.pic_gb = round(max(self._pic_carte_gb - self._base_gb, torche), 2)
            self.occupation_pct = round(100.0 * self._pic_carte_gb / self.carte_gb, 1)
            self.portee = "delta_carte"
        elif torche > 0.0:
            self.pic_gb, self.portee = round(torche, 2), "pytorch"
        return None


def _pic_torch() -> float:
    """Ce que PyTorch a RÉSERVÉ au maximum, en Go.

    « Réservé » et non « alloué » : l'allocateur garde ses blocs, et c'est
    bien cette réservation-là qui prive les bibliothèques tierces — c'est le
    chiffre qui a fait tomber l'occlusion de 27 à 6 secondes une fois rendu.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved() / 1024 ** 3
    except Exception:                                            # noqa: BLE001
        pass
    return 0.0
