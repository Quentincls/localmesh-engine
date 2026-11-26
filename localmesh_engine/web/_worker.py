"""Run one generation in a process that releases CUDA memory on exit."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..progress import english_stage
from ..run_history import SIDES, RunHistory
from .._locking import LockBusy, process_lock


def generate(history: RunHistory, run_id: str) -> None:
    from .. import config
    from ..pipeline import Engine, GenerateSettings

    row = history.get(run_id)
    if row is None or row["origin"] != "web" or row["status"] != "running":
        raise ValueError("Generation request was not found.")

    gpu = config.gpu_report()
    print(f"GPU: {gpu.name}, {gpu.total_vram_gb:.1f} GB", flush=True)
    if not config.palier_tenable(row["tier"], gpu.total_vram_gb):
        raise RuntimeError(f"The {row['tier']} tier requires more video memory than this GPU has.")

    photos = history.source_paths(run_id)
    side_views = {internal: photos[role] for role, internal in SIDES.items() if role in photos}
    settings = GenerateSettings(
        images=[photos["front"]], vues=side_views, detail=row["tier"],
        seed=row["seed"], projection_texture=row["projection"],
    )

    def advance(stage: str, fraction: float) -> None:
        display = english_stage(stage)
        history.update_run(run_id, stage=display, progress=fraction)
        print(f"{fraction:6.1%}  {display}", flush=True)

    result = Engine().generate(settings, history.result_dir(run_id), progress=advance)
    print(f"{result.faces:,} faces, {result.duration_s:.0f} s, "
          f"peak {result.peak_vram_gb:.2f} GB, seed {result.seed}", flush=True)
    history.record_result(run_id, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args()
    history = RunHistory(args.runtime)

    try:
        with process_lock(history.root / ".web-worker.lock"):
            row = history.get(args.run_id)
            if row is None or row["origin"] != "web" or row["status"] != "queued":
                raise ValueError("Queued generation request was not found.")
            history.update_run(args.run_id, status="running", stage="Starting generation")
            try:
                generate(history, args.run_id)
            except KeyboardInterrupt:
                history.update_run(args.run_id, status="interrupted", stage="Interrupted",
                                   error="Generation was interrupted.")
                return 1
            except Exception as exc:
                logging.exception("Generation failed")
                history.update_run(args.run_id, status="failed", stage="Failed",
                                   error=str(exc) or type(exc).__name__)
                return 1
    except LockBusy:
        print("A LocalMesh generation is already running.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
