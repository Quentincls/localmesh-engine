"""Ce que le paquet ComfyUI fournissait, et que LumenGen fournit autrement.

La copie de TRELLIS.2 distribuee par le wrapper ComfyUI importe `folder_paths`
et `comfy.utils` directement dans la bibliotheque — pas seulement dans les
noeuds. Trois chemins de modeles et une barre de progression, rien de plus,
mais cela suffisait a rendre le paquet inutilisable hors de ComfyUI.

## Pourquoi ca se resout tout seul

Une premiere version demandait a l'appelant de renseigner `racine_modeles`
avant le premier import. Mauvaise idee : le paquet est charge sous un alias
(`lumengen.t2pkg.trellis2`), donc le renseigner par un autre chemin d'import
donne un OBJET DIFFERENT, et le pipeline ne voit rien. L'erreur ne se
manifeste qu'au chargement des poids, loin de sa cause.

Alors ce module ne demande plus rien : il resout la racine comme le fait
`config.py`, depuis `LUMENGEN_ROOT`. Un seul endroit de verite, aucun ordre
d'appel a respecter.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Renseignable a la main si besoin, mais ce n'est plus necessaire.
racine_modeles = None


def modeles() -> Path:
    if racine_modeles is not None:
        return Path(racine_modeles)

    # UNE SEULE SOURCE DE VERITE, ET C'EST CELLE DU PAQUET.
    #
    # Ce fichier ne lisait que LUMENGEN_ROOT. Or `config.py` accepte les deux
    # noms ET SON MESSAGE D'ERREUR ORDONNE DE POSER LOCALMESH_ROOT : quelqu'un
    # qui suivait cette consigne voyait le demarrage reussir, puis le
    # chargement des poids echouer cent lignes plus loin en nommant une
    # variable dont la documentation publique ne parle pas. Le piege etait
    # invisible ici parce que l'application privee pose toujours l'ancien nom.
    try:
        from ...config import MODELS_ROOT
        return Path(MODELS_ROOT)
    except Exception:                                         # noqa: BLE001
        pass

    racine = (os.environ.get("LOCALMESH_ROOT")
              or os.environ.get("LUMENGEN_ROOT"))
    if racine:
        return Path(racine) / "models"

    # Copie portable : le runtime est pose a cote du moteur. La remontee etait
    # calee sur l'ANCIEN emplacement de ce fichier (trellis2 -> _wheels_src ->
    # engine) ; depuis `_vendu_trellis2` elle designait un dossier que rien ne
    # cree. On essaie les profondeurs plausibles plutot qu'une seule fausse.
    ici = Path(__file__).resolve()
    for niveau in (3, 4, 5):
        if niveau < len(ici.parents):
            a_cote = ici.parents[niveau] / "models"
            if a_cote.is_dir():
                return a_cote

    raise RuntimeError(
        "Impossible de situer le dossier des modeles : ni `racine_modeles`, "
        "ni LOCALMESH_ROOT, ni LUMENGEN_ROOT, ni un dossier `models` a cote "
        "du moteur."
    )
