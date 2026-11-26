# -*- coding: utf-8 -*-
"""Generate a textured 3D model from one or four photos on the command line.

    python -m localmesh_engine photo.png
    python -m localmesh_engine front.png --right r.png --left l.png --back b.png
    python -m localmesh_engine photo.png --tier high --seed 42 --to out/
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from .progress import english_stage
from .run_history import SIDES, TIERS, RunHistory


def main(argv: list[str] | None = None) -> int:
    # REGARDER AVANT D'INSTALLER. `import localmesh_engine` monte la
    # configuration, qui exige la racine du runtime : sans elle, meme
    # `--help` mourait sur une exception, avant argparse. Quelqu'un qui
    # decouvre le depot ne pouvait donc pas lire l'aide sans avoir d'abord
    # pose plusieurs gigaoctets de poids.
    p = argparse.ArgumentParser(
        prog="localmesh_engine",
        description="Generate a textured 3D model from one or four photos of a subject.")
    p.add_argument("photo", type=Path,
                   help="front photo, which guides the shape of the model")
    # CHAQUE OPTION PORTE SON NOM ANGLAIS A COTE DU FRANCAIS. Le code est
    # ecrit en francais, les options l'etaient aussi, et un lecteur anglophone
    # se cognait sur `--dos` sans deviner qu'il s'agit du dos. Les deux
    # graphies marchent. `dest` est fige : sans lui, argparse prendrait le
    # PREMIER nom long comme attribut et tout ce qui suit lirait `a.right`.
    p.add_argument("--right", "--droite", dest="droite", type=Path, metavar="PHOTO",
                   help="right-side photo of the subject")
    p.add_argument("--left", "--gauche", dest="gauche", type=Path, metavar="PHOTO",
                   help="left-side photo of the subject")
    p.add_argument("--back", "--dos", dest="dos", type=Path, metavar="PHOTO",
                   help="back photo of the subject")
    p.add_argument("--tier", "--palier", dest="palier", default="standard",
                   choices=TIERS,
                   help="quality tier (see docs/RECIPES.md)")
    p.add_argument("--seed", "--graine", dest="graine", type=int, default=-1,
                   metavar="SEED", help="random seed (use -1 to choose one automatically)")
    p.add_argument("--to", "--vers", dest="vers", type=Path, default=Path("."),
                   metavar="DIRECTORY", help="directory for generated files")
    p.add_argument("--no-projection", "--sans-projection", dest="projection",
                   action="store_false",
                   help="keep the generated texture without projecting the source photos")
    a = p.parse_args(argv)

    if not a.photo.is_file():
        print("Front photo not found: %s" % a.photo, file=sys.stderr)
        return 2

    sources = {"front": (a.photo.name, a.photo.resolve())}
    for role, chemin in (("right", a.droite), ("left", a.gauche), ("back", a.dos)):
        if chemin is None:
            continue
        if not chemin.is_file():
            print("%s view not found: %s" % (role.capitalize(), chemin), file=sys.stderr)
            return 2
        sources[role] = (chemin.name, chemin.resolve())

    if len(sources) not in (1, 4):
        # Le multivue est mesuré à quatre vues, pas à deux ou trois. Mieux
        # vaut le dire que de rendre un objet dont personne ne sait ce qu'il
        # vaut.
        print("Multi-view generation requires all three side photos: --right, --left, --back.",
              file=sys.stderr)
        return 2

    # Keep help and input validation independent of CUDA and the model weights.
    from . import config

    carte = config.gpu_report()
    print("GPU: %s, %.1f GB VRAM" % (carte.name, carte.total_vram_gb), file=sys.stderr)
    if not config.palier_tenable(a.palier, carte.total_vram_gb):
        print("The %s tier requires more VRAM than this GPU has."
              % a.palier, file=sys.stderr)
        return 3

    sortie = a.vers.resolve()
    sortie.mkdir(parents=True, exist_ok=True)

    history = RunHistory(config.RUNTIME_ROOT)
    run = history.create_run(origin="cli", sources=sources, tier=a.palier,
                             seed=a.graine, projection=a.projection)
    run_id = run["id"]
    depart = time.time()
    with (history.cli_lock(run_id),
          history.log_path(run_id).open("w", encoding="utf-8", buffering=1) as log):
        print("GPU: %s, %.1f GB VRAM" % (carte.name, carte.total_vram_gb), file=log)
        history.update_run(run_id, status="running", stage="Starting generation")

        def report(line: str) -> None:
            print(line, file=sys.stderr, flush=True)
            print(line, file=log)

        def avance(etape: str, part: float) -> None:
            stage = english_stage(etape)
            report("  %5.1f%%  %s" % (part * 100, stage))
            history.update_run(run_id, stage=stage, progress=part)

        try:
            from .pipeline import Engine, GenerateSettings

            photos = history.source_paths(run_id)
            vues = {SIDES[role]: path for role, path in photos.items() if role in SIDES}
            reglages = GenerateSettings(images=[photos["front"]], vues=vues,
                                        detail=a.palier, seed=a.graine,
                                        projection_texture=a.projection)
            resultat = Engine().generate(reglages, history.result_dir(run_id), progress=avance)
            history.record_result(run_id, resultat)
        except KeyboardInterrupt:
            history.update_run(run_id, status="interrupted", stage="Interrupted")
            report("Generation interrupted.")
            return 130
        except Exception as exc:
            history.update_run(run_id, status="failed", stage="Failed", error=str(exc))
            traceback.print_exc(file=log)
            raise

        try:
            output = history.export_run(run_id, sortie)
        except OSError as exc:
            archived = history.asset(run_id, "model")
            error = f"Export failed. The generated model is saved at {archived}: {exc}"
            history.update_run(run_id, error=error)
            report(error)
            return 1

        print(output)
        report("%d faces, %.0f s, peak VRAM %.2f GB, seed %d"
               % (resultat.faces, time.time() - depart,
                  resultat.peak_vram_gb or 0.0, resultat.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
