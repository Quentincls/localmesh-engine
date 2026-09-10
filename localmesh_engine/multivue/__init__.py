# -*- coding: utf-8 -*-
"""Quatre photos d'un objet, un seul volume.

C'est ce que le moteur apporte et que le modèle d'origine n'a pas : sa page
n'offre qu'une image, et l'ancienne façon de fondre plusieurs vues, en
pondérant par la position supposée de chacune, se trompait dès que le modèle
plaçait l'objet autrement qu'attendu. Un feu tricolore sortait avec une
lentille de plus à l'arrière ; un samouraï, avec deux lames.

**Ce chemin fait autrement.** Chaque vue est projetée sur la grille par sa
caméra, et c'est un modèle entraîné pour ça qui fusionne. La pose de l'objet
cesse d'être une supposition.

La séquence, en sept temps :

1. les quatre photos sont détourées comme la voie à une photo ;
2. la latéralité est décidée, ou imposée par l'appelant ;
3. les vues sont encodées en 512 ;
4. la structure creuse multivue en sort un volume de cellules ;
5. la forme descend en cascade, 512 puis 1024 ;
6. le maillage est décodé ;
7. texture, nettoyage et finition sont ceux de la voie normale, inchangés.

Rien de tout cela ne demande plus de mémoire vive que la voie à une photo :
les poids se chargent l'un après l'autre et la carte est rendue entre deux
étapes. Ce qui coûte, c'est la place sur le disque.

Ce module ne télécharge rien. Les poids supplémentaires vivent sous
`config.MODELS_ROOT / "multivue"`, et leur absence se dit clairement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

#: Les quatre rôles, dans l'ordre où les modèles les attendent.
ROLES = ("front", "right", "back", "left")

#: Le vocabulaire de l'application, et sa traduction. La face est l'image
#: principale ; les trois autres arrivent nommées.
DEPUIS_LE_PRODUIT = {"droite": "right", "gauche": "left", "dos": "back"}

Lateralite = Literal["auto", "normale", "inversee"]


@dataclass
class Poids:
    """Où sont les modèles supplémentaires, et lesquels manquent.

    Le moteur ne va rien chercher : c'est l'hôte qui télécharge, et lui qui
    décide comment le dire à quelqu'un. Ici on se contente de savoir.
    """

    racine: Path
    structure: Path
    forme_512: Path
    forme_1024: Path
    champ: Path
    cameras: Optional[Path] = None

    @classmethod
    def depuis(cls, models_root: Path) -> "Poids":
        r = Path(models_root) / "multivue"
        cameras = r / "cameras" / "model.safetensors"
        return cls(
            racine=r,
            structure=r / "structure_mv_fp8.safetensors",
            forme_512=r / "forme_512_mv_fp8.safetensors",
            forme_1024=r / "forme_1024_mv_fp8.safetensors",
            champ=r / "champ.safetensors",
            cameras=cameras if cameras.is_file() else None,
        )

    def manquants(self) -> list[str]:
        """Ce qui n'est pas là, en clair, dans l'ordre où on s'en sert."""
        absents = []
        # Les trois modèles de flux se construisent depuis leur description ;
        # sans elle, le fichier de poids ne sert à rien. Le réseau de champ,
        # lui, se construit avec ses valeurs par défaut : il n'en a pas.
        for nom, chemin in (("structure multivue", self.structure),
                            ("forme 512", self.forme_512),
                            ("forme 1024", self.forme_1024)):
            if not chemin.is_file():
                absents.append(nom)
            elif not chemin.with_suffix(".json").is_file():
                absents.append(nom + " (sa description)")
        if not self.champ.is_file():
            absents.append("réseau de champ")
        return absents

    def pretes(self) -> bool:
        return not self.manquants()


class MultivueIndisponible(RuntimeError):
    """La brique multivue n'est pas installée, et on le dit avant de commencer.

    Elle porte la liste de ce qui manque : une application qui propose un
    téléchargement a besoin de savoir quoi, pas seulement que.
    """

    def __init__(self, manquants: list[str]):
        self.manquants = manquants
        super().__init__(
            "The multi-view module is not installed. Missing: "
            + ", ".join(manquants))


def roles_depuis_vues(vues: dict) -> dict:
    """Traduire les noms de l'application vers ceux des modèles.

    On refuse un jeu incomplet plutôt que de compléter en silence : la recette
    est mesurée à quatre vues, et rendre un objet à trois en le présentant
    comme un multivue serait mentir sur ce qu'il vaut.
    """
    inconnus = sorted(set(vues) - set(DEPUIS_LE_PRODUIT))
    if inconnus:
        raise ValueError("Unknown view names: %s" % ", ".join(inconnus))
    manquants = sorted(set(DEPUIS_LE_PRODUIT) - set(vues))
    if manquants:
        raise ValueError(
            "Multi-view needs all three sides; missing: %s" % ", ".join(manquants))
    return {DEPUIS_LE_PRODUIT[k]: v for k, v in vues.items()}


def parite(lateralite: Lateralite, decision_mesuree: Optional[str]) -> dict:
    """À l'endroit, ou les deux profils échangés.

    POURQUOI CE N'EST PAS QU'UNE ÉTIQUETTE. On a d'abord cru que l'interface
    suffisait : elle demande à l'utilisateur de ranger chaque photo dans un
    emplacement nommé, donc elle sait. Elle ne sait pas. Une étiquette dit où
    la photo a été rangée, pas d'où elle a été prise. Sur le samouraï de
    référence, le modèle de caméras mesure la vue rangée dans « droite » à
    +98 degrés de la face et celle rangée dans « gauche » à -54 degrés, qui
    n'est même pas un profil. Ce sont deux angles, pas deux avis.

    D'où les trois réponses : mesurer quand on peut, imposer quand l'appelant
    sait, et ne jamais deviner en silence.
    """
    if lateralite == "normale":
        return {"swap_sides": False, "source": "imposee"}
    if lateralite == "inversee":
        return {"swap_sides": True, "source": "imposee"}
    if decision_mesuree in ("nominal", "swap_sides"):
        return {"swap_sides": decision_mesuree == "swap_sides", "source": "mesuree"}
    # AUCUNE MESURE : ON PREND LA CONVENTION, ET ON LE DIT.
    #
    # Refuser le travail serait le pire des deux mondes : l'utilisateur a
    # quatre photos et n'obtient rien, pour un choix binaire qu'il peut
    # corriger en un mot. Sur les six sujets mesures, cinq suivent cette
    # convention. Le resultat porte la mention, pour que celui dont les
    # profils sortent inverses sache quoi changer plutot que de chercher.
    return {"swap_sides": False, "source": "convention"}
