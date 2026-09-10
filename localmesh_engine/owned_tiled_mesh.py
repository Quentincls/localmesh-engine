"""Sparse mesh extraction with one owner per dual-grid quad.

Halo cells provide neighbors, but only core cells may emit their intersected
edges. This prevents overlapping tiles from emitting the same oriented faces.
Uses the installed extractor without modifying the installed runtime.
"""
import os
import torch
from o_voxel.convert import flexible_dual_grid_to_mesh
from o_voxel.convert import tiled_flexible_dual_grid_to_mesh as _legacy_tiled_mesh


@torch.no_grad()
def tiled_flexible_dual_grid_to_mesh(
    coords, dual_vertices, intersected_flag, split_weight,
    aabb, grid_size, tile_size=128, overlap=1, train=False,
):
    if os.environ.get('LUMENGEN_FIX_TILED_MESH', '1') == '0':
        return _legacy_tiled_mesh(
            coords, dual_vertices, intersected_flag, split_weight,
            aabb=aabb, grid_size=grid_size, tile_size=tile_size,
            overlap=overlap, train=train,
        )
    if tile_size < 1 or overlap < 1:
        raise ValueError('Positive tile size and at least one halo cell required')
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('Expected sparse coordinates [N,3]')
    if not len(coords):
        return None, None
    device=coords.device
    lo=coords.min(dim=0).values.cpu().tolist()
    hi=coords.max(dim=0).values.cpu().tolist()
    all_vertices,all_faces=[],[]
    offset=0
    for x in range(lo[0],hi[0]+1,tile_size):
        for y in range(lo[1],hi[1]+1,tile_size):
            for z in range(lo[2],hi[2]+1,tile_size):
                lower=torch.tensor([x,y,z],device=device,dtype=coords.dtype)
                upper=lower+tile_size
                halo=((coords>=lower-overlap)&(coords<upper+overlap)).all(dim=1)
                if not halo.any():
                    continue
                selected=coords[halo].contiguous()
                core=((selected>=lower)&(selected<upper)).all(dim=1)
                flags=(intersected_flag[halo]&core[:,None]).contiguous()
                if not flags.any():
                    continue
                weights=split_weight[halo].contiguous() if split_weight is not None else None
                try:
                    vertices,faces=flexible_dual_grid_to_mesh(
                        selected,dual_vertices[halo].contiguous(),flags,weights,
                        aabb=aabb,grid_size=grid_size,train=train,
                    )
                except RuntimeError as exc:
                    # A failed tile is an incomplete object; never silently omit it.
                    raise RuntimeError(f'Mesh extraction failed in tile {(x,y,z)}') from exc
                if vertices is not None and len(vertices):
                    all_vertices.append(vertices)
                    all_faces.append(faces+offset)
                    offset+=len(vertices)
    if not all_vertices:
        return None,None
    vertices,inverse=torch.unique(torch.cat(all_vertices),dim=0,return_inverse=True)
    return vertices,inverse[torch.cat(all_faces)]
