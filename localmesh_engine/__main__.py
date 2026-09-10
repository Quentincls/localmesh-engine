# -*- coding: utf-8 -*-
"""Une photo, ou quatre, et un objet 3D — depuis la ligne de commande.

    python -m localmesh_engine photo.png
    python -m localmesh_engine face.png --droite d.png --gauche g.png --dos b.png
    python -m localmesh_engine photo.png --palier high --graine 42 --vers sortie/

C'est le chemin le plus court pour essayer le moteur sans écrire une ligne de
Python. Il ne range rien dans une bibliothèque et n'ouvre aucune fenêtre.
Il ne télécharge pas les gros modèles non plus — mais il n'est PAS hors
ligne : le détourage va chercher son modèle sur le Hub à la première
génération s'il n'est pas déjà en cache. Les poids doivent être en place
sous la racine du runtime (`LOCALMESH_ROOT`, ou `LUMENGEN_ROOT` pour les
installations d'avant), et l'objet sort là où on le demande.

La progression s'écrit sur la sortie d'erreur, une ligne par étape, pour
qu'on puisse rediriger le résultat sans que le compte-rendu s'y mélange.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # REGARDER AVANT D'INSTALLER. `import localmesh_engine` monte la
    # configuration, qui exige la racine du runtime : sans elle, meme
    # `--help` mourait sur une exception, avant argparse. Quelqu'un qui
    # decouvre le depot ne pouvait donc pas lire l'aide sans avoir d'abord
    # pose plusieurs gigaoctets de poids.
    p = argparse.ArgumentParser(
        prog="localmesh_engine",
        description="Génère un objet 3D texturé à partir d'une photo, "
                    "ou de quatre côtés du même sujet.")
    p.add_argument("photo", type=Path,
                   help="la photo de face ; c'est elle qui porte la forme")
    p.add_argument("--droite", type=Path, help="le côté droit du sujet")
    p.add_argument("--gauche", type=Path, help="le côté gauche du sujet")
    p.add_argument("--dos", type=Path, help="le dos du sujet")
    p.add_argument("--palier", default="standard",
                   choices=["draft", "standard", "high", "max"],
                   help="la recette de qualité (voir RECETTES.md)")
    p.add_argument("--graine", type=int, default=-1,
                   help="la graine ; -1 en tire une au hasard")
    p.add_argument("--vers", type=Path, default=Path("."),
                   help="le dossier où déposer l'objet")
    a = p.parse_args(argv)

    if not a.photo.is_file():
        print("photo introuvable : %s" % a.photo, file=sys.stderr)
        return 2

    # L'import est ici, pas en tête : charger le moteur prend plusieurs
    # secondes, et `--help` ne doit pas les payer.
    from . import config
    from .pipeline import Engine, GenerateSettings

    vues: dict[str, Path] = {}
    for nom, chemin in (("droite", a.droite), ("gauche", a.gauche), ("dos", a.dos)):
        if chemin is None:
            continue
        if not chemin.is_file():
            print("vue %s introuvable : %s" % (nom, chemin), file=sys.stderr)
            return 2
        vues[nom] = chemin

    if vues and len(vues) != 3:
        # Le multivue est mesuré à quatre vues, pas à deux ou trois. Mieux
        # vaut le dire que de rendre un objet dont personne ne sait ce qu'il
        # vaut.
        print("le multivue attend les TROIS côtés : --droite, --gauche, --dos",
              file=sys.stderr)
        return 2

    carte = config.gpu_report()
    print("carte : %s, %.1f Go" % (carte.name, carte.total_vram_gb), file=sys.stderr)
    if not config.palier_tenable(a.palier, carte.total_vram_gb):
        print("le palier %s demande plus de mémoire que cette carte n'en a"
              % a.palier, file=sys.stderr)
        return 3

    sortie = a.vers.resolve()
    sortie.mkdir(parents=True, exist_ok=True)

    depart = time.time()

    def avance(etape: str, part: float) -> None:
        print("  %5.1f%%  %s" % (part * 100, etape), file=sys.stderr, flush=True)

    moteur = Engine()
    reglages = GenerateSettings(images=[a.photo.resolve()], vues=vues,
                                detail=a.palier, seed=a.graine)
    resultat = moteur.generate(reglages, sortie, progress=avance)

    print("%s" % resultat.glb_path)
    print("%d faces, %.0f s, pic %.2f Go, graine %d"
          % (resultat.faces, time.time() - depart,
             resultat.peak_vram_gb or 0.0, resultat.seed), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
