"""Per-stage previews, so a long generation is not a blank wait.

Two milestones, both taken from data the pipeline already produces:

* **structure** - the coarse occupancy grid, available about a tenth of the way
  in and costing nothing to export. If the silhouette is wrong, the artist can
  cancel here instead of finding out twenty minutes later.
* **shape** - the real geometry, decoded before texturing. Costs one extra
  decode (a few seconds, done at low resolution regardless of the final target)
  and shows exactly what will be textured.

Both are untextured on purpose: they are for judging form, and pretending
otherwise would just mean waiting longer for a worse version of the final.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("localmesh_engine.previews")

#: Decoding the shape latent for a preview is done here rather than at the
#: final resolution: 512 is plenty to judge a silhouette and keeps the detour
#: to a few seconds even when the job targets 1536.
PREVIEW_DECODE_RESOLUTION = 512


def voxel_preview(coords: torch.Tensor, grid_size: int, out_path: Path) -> Optional[Path]:
    """Turn the sparse occupancy grid into a blocky mesh.

    `coords` is [N, 4]: batch index, then x/y/z. Marching cubes over the
    occupancy volume gives the stair-stepped surface one expects from voxels,
    in a few milliseconds - far cheaper than instancing N boxes.
    """
    try:
        import trimesh
        from skimage import measure

        c = coords.detach().cpu().numpy()
        xyz = c[:, 1:] if c.shape[1] == 4 else c
        if xyz.shape[0] == 0:
            return None

        volume = np.zeros((grid_size + 2,) * 3, dtype=np.float32)
        idx = np.clip(xyz.astype(int), 0, grid_size - 1) + 1
        volume[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

        verts, faces, _normals, _ = measure.marching_cubes(volume, level=0.5)

        # Same frame as the final asset: unit cube centred on the origin.
        verts = (verts - 1.0) / grid_size - 0.5

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        log.info("structure preview: %d voxels -> %d faces", xyz.shape[0], len(faces))
        return out_path
    except Exception as exc:
        # A preview is a nicety; never let it take down a generation.
        log.warning("structure preview failed: %s", exc)
        return None


def shape_preview(pipeline, shape_slat, out_path: Path,
                  resolution: int = PREVIEW_DECODE_RESOLUTION) -> Optional[Path]:
    """Decode the shape latent to an untextured mesh."""
    try:
        import trimesh

        meshes, _subs = pipeline.decode_shape_slat(shape_slat, resolution)
        mesh = meshes[0]
        verts = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        if len(faces) == 0:
            return None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out_path)
        log.info("shape preview: %d faces at resolution %d", len(faces), resolution)
        return out_path
    except Exception as exc:
        log.warning("shape preview failed: %s", exc)
        return None
    finally:
        torch.cuda.empty_cache()
