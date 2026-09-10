"""Bake the dense mesh's relief into a tangent-space normal map for the light one.

Why this exists
---------------
The poly budget throws away real geometry, and with it every fine detail: the
chasing on a crown, the grain on a rock. Nothing puts it back. A normal map
does not restore the geometry - it restores how the surface *responds to light*,
which is what the eye actually reads at anything other than a silhouette.

This is the standard high-to-low bake from game art, done on the GPU:

1. rasterise the low mesh **in UV space**, so every texel knows its world
   position and its own tangent frame;
2. from each texel, cast a ray along the low normal at the dense mesh;
3. express the dense mesh's normal at the hit in the low mesh's tangent frame.

What it cannot do is fix a silhouette - see the silhouette check for that.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("localmesh_engine.normalbake")

#: Rays start slightly outside the low surface and travel inwards, so a texel
#: sitting exactly on the dense surface does not self-hit at distance zero.
#: Expressed as a fraction of the model's bounding-box diagonal.
_CAGE_OUT = 0.008
_RAY_LENGTH = 0.015

#: A hit whose surface faces away from the low-poly normal is not "our" surface:
#: on thin or openwork geometry the ray has punched through and landed on the
#: back of the shell, or on a different filigree element entirely. Those hits
#: produce the violent greens and oranges that betray a bad bake, so they are
#: rejected in favour of a flat normal.
_MIN_NORMAL_AGREEMENT = 0.15


@dataclass
class BakeReport:
    size: int
    coverage: float          # fraction of texels that found the dense surface
    faces_high: int
    faces_low: int
    #: Hits discarded because the ray went through a thin wall. A high value
    #: means the object is openwork and the map will be partly flat - which is
    #: correct, and better than confetti.
    pierced: float = 0.0

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "coverage_pct": round(self.coverage * 100, 1),
            "pierced_pct": round(self.pierced * 100, 1),
            "faces_high": self.faces_high,
            "faces_low": self.faces_low,
        }


def _vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    tri = v[f.long()]
    face_n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    out = torch.zeros_like(v)
    out.index_add_(0, f.reshape(-1).long(), face_n.repeat_interleave(3, dim=0))
    return torch.nn.functional.normalize(out, dim=-1, eps=1e-12)


def _vertex_tangents(v: torch.Tensor, f: torch.Tensor, uv: torch.Tensor,
                     n: torch.Tensor) -> torch.Tensor:
    """Per-vertex tangents from the UV parameterisation (Lengyel's method).

    The tangent must come from the UVs, not from an arbitrary axis: it is the
    direction in which U increases across the surface, and it is what makes a
    baked map readable by any other renderer.
    """
    idx = f.long()
    tri_v = v[idx]
    tri_uv = uv[idx]

    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    d1 = tri_uv[:, 1] - tri_uv[:, 0]
    d2 = tri_uv[:, 2] - tri_uv[:, 0]

    denom = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    # Degenerate UV triangles (zero area in texture space) contribute nothing
    # rather than infinities.
    r = torch.where(denom.abs() > 1e-12, 1.0 / denom, torch.zeros_like(denom))
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]

    acc = torch.zeros_like(v)
    acc.index_add_(0, idx.reshape(-1), tangent.repeat_interleave(3, dim=0))

    # Gram-Schmidt against the normal, so T is exactly perpendicular to N.
    acc = acc - n * (n * acc).sum(-1, keepdim=True)
    return torch.nn.functional.normalize(acc, dim=-1, eps=1e-12)


def _barycentric(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                 c: torch.Tensor) -> torch.Tensor:
    """Barycentric coordinates of p inside triangle abc, per row."""
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    denom = torch.where(denom.abs() > 1e-16, denom, torch.ones_like(denom))
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return torch.stack([1.0 - v - w, v, w], dim=-1)


def _sample(array: np.ndarray, uv: torch.Tensor, dev) -> Optional[torch.Tensor]:
    """Bilinear lookup of `array` (HxWxC, uint8 or float) at UV coordinates.

    UV origin is bottom-left, image row 0 is the top, hence the V flip.
    """
    if array is None or array.ndim != 3:
        return None
    tex = torch.as_tensor(np.ascontiguousarray(array), device=dev)
    if tex.dtype != torch.float32:
        tex = tex.float()
    if tex.max() > 1.5:
        tex = tex / 255.0

    grid = torch.stack([uv[:, 0] * 2 - 1, (1.0 - uv[:, 1]) * 2 - 1], dim=-1)
    out = torch.nn.functional.grid_sample(
        tex.permute(2, 0, 1).unsqueeze(0),
        grid.view(1, 1, -1, 2),
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return (out.squeeze(0).squeeze(1).T.clamp(0, 1) * 255).round().to(torch.uint8)


#: Bilinear filtering reaches one texel past an island edge, mip-mapping much
#: further. This many pixels of true per-island padding covers both.
_PADDING = 12


def _fill_gutters(rgb: np.ndarray, mask: np.ndarray,
                  padding: int = _PADDING) -> np.ndarray:
    """Pad each UV island outwards with its own colour, then fill the rest.

    The distinction matters. A UV unwrap of a generated mesh produces hundreds
    of small islands packed close together, and two islands adjacent *in the
    atlas* usually come from unrelated places on the model. Filling a gutter
    texel with the globally nearest filled texel therefore paints one island's
    colour right next to another's - and bilinear filtering then drags it back
    across the seam, giving the model a cracked, flaking look along every one
    of those hundreds of boundaries.

    So the first `padding` pixels are grown from each island's *own* edge, one
    ring at a time. Only the far gutters, which no filter ever reaches, are
    then closed with a nearest-neighbour pass - needed because an empty alpha
    there would otherwise read as a stencil cutout.
    """
    if mask.all() or not mask.any():
        return rgb
    plat = rgb.reshape(-1, rgb.shape[2])
    return plat[carte_gouttieres(mask, padding)].reshape(rgb.shape)


#: La dernière carte calculée, et le masque qui l'a produite. Une seule
#: entrée : une conversion n'utilise qu'un masque, et en garder plus ne
#: retiendrait que des mégaoctets pour rien.
_cache_gouttieres = None


def carte_gouttieres(mask: np.ndarray, padding: int = _PADDING) -> np.ndarray:
    """Où chaque texel va chercher sa couleur. Calculée une fois par masque.

    Rien de ce remplissage ne dépend de l'IMAGE : ni la croissance anneau par
    anneau depuis le bord de chaque îlot, ni la passe de plus proche voisin
    qui ferme les gouttières lointaines. Tout ne dépend que de la couverture.
    Or une conversion appelle ce remplissage CINQ fois — couleur, normales,
    occlusion, albédo, matières — sur le même masque, et recalculait donc
    cinq fois la même chose, transformée de distance comprise : 1,8 s à
    chaque coup.

    On déplace donc des INDICES au lieu de couleurs. La suite des « ce texel
    prend celui-là » est exactement la même, et l'appliquer à une image
    devient une lecture indexée. Composer les deux étapes est exact : lire
    des indices puis lire des valeurs revient à lire les valeurs par les
    indices composés.
    """
    global _cache_gouttieres

    cle = (mask.shape, padding, hash(mask.tobytes()))
    if _cache_gouttieres is not None and _cache_gouttieres[0] == cle:
        return _cache_gouttieres[1]

    h, w = mask.shape
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    filled = mask.copy()

    for _ in range(padding):
        holes = ~filled
        if not holes.any():
            break
        newly = np.zeros_like(filled)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src_idx = np.roll(np.roll(idx, dy, 0), dx, 1)
            src_ok = np.roll(np.roll(filled, dy, 0), dx, 1)
            take = holes & src_ok & ~newly
            if take.any():
                idx[take] = src_idx[take]
                newly |= take
        if not newly.any():
            break
        filled |= newly

    if not filled.all():
        from scipy import ndimage

        nearest = ndimage.distance_transform_edt(
            ~filled, return_distances=False, return_indices=True)
        idx = idx[tuple(nearest)]

    _cache_gouttieres = (cle, idx.reshape(-1))
    return _cache_gouttieres[1]


def repad_atlas(vertices: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor,
                textures: dict, size: Optional[int] = None) -> dict:
    """Re-pad each UV island with its own colour.

    The atlas that comes out of to_glb has its islands filled correctly but its
    *gutters* filled with speckle - scattered cyan, black and lavender texels.
    Bilinear filtering and mip-mapping reach across island edges, so that noise
    lands on the model as flecks and torn-looking patches along every seam, and
    there are hundreds of seams.

    Rasterising the mesh's own UVs gives the exact island coverage, after which
    the same padding used for baked maps applies. Nothing inside an island is
    touched.
    """
    from . import uvraster

    if uv is None or not textures:
        return {}

    dev = torch.device("cuda")
    uv = uv.to(dev, torch.float32)
    faces = faces.to(dev, torch.int64)
    if uv.shape[0] != vertices.shape[0]:
        log.warning("cannot re-pad: %d UVs for %d vertices",
                    uv.shape[0], vertices.shape[0])
        return {}

    first = next(iter(textures.values()))
    size = size or int(first.shape[0])

    _, covered = uvraster.atlas(uv, faces, size)

    if not covered.any():
        return {}

    out = {}
    for name, array in textures.items():
        if array is None or array.ndim != 3 or array.shape[0] != size:
            continue
        out[name] = _fill_gutters(np.ascontiguousarray(array), covered)
    log.info("re-padded %d atlas map(s), islands cover %.1f%%",
             len(out), 100 * float(covered.mean()))
    return out


def bake(
    high_vertices: torch.Tensor,
    high_faces: torch.Tensor,
    low_vertices: torch.Tensor,
    low_faces: torch.Tensor,
    low_uv: torch.Tensor,
    size: int = 2048,
    high_uv: Optional[torch.Tensor] = None,
    transfer: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], Optional[BakeReport], dict]:
    """Bake a normal map, and optionally re-project the dense mesh's textures.

    `transfer` maps a name to an HxWx3 or HxWx4 float array addressed by
    `high_uv`. Each one is resampled into the low mesh's own UV layout.

    This second part is not a nicety. Re-unwrapping a mesh invalidates every
    texture that was addressed by the old parameterisation: the maps still
    exist, still look like textures, and now paint completely the wrong parts
    of the model. Whenever the topology changes, the maps must come with it.
    """
    import cumesh

    from . import uvraster

    if low_uv is None:
        log.warning("normal bake skipped: the low mesh has no UVs")
        return None, None, {}

    # CuMesh does not guarantee which device its outputs land on, and
    # nvdiffrast fails with an opaque CUDAGuardImpl error rather than saying
    # "this tensor is on the CPU". Normalise once, here.
    dev = torch.device("cuda")
    high_vertices = high_vertices.to(dev, torch.float32).contiguous()
    high_faces = high_faces.to(dev, torch.int32).contiguous()
    low_vertices = low_vertices.to(dev, torch.float32).contiguous()
    low_faces = low_faces.to(dev, torch.int32).contiguous()
    low_uv = low_uv.to(dev, torch.float32).contiguous()

    if low_uv.shape[0] != low_vertices.shape[0]:
        log.warning("normal bake skipped: %d UVs for %d vertices",
                    low_uv.shape[0], low_vertices.shape[0])
        return None, None, {}

    extent = float((high_vertices.max(0).values - high_vertices.min(0).values).norm())
    cage_out = extent * _CAGE_OUT
    ray_len = extent * _RAY_LENGTH

    low_n = _vertex_normals(low_vertices, low_faces)
    low_t = _vertex_tangents(low_vertices, low_faces, low_uv, low_n)
    high_n = _vertex_normals(high_vertices, high_faces)

    # --- 1. rasterise the low mesh in UV space -----------------------------
    # Comes back top-down, which is the convention everything downstream uses;
    # the caller no longer flips anything by hand.
    (pos, nrm, tan), covered = uvraster.atlas(
        low_uv, low_faces, size, [low_vertices, low_n, low_t], as_numpy=False)
    if not bool(covered.any()):
        log.warning("normal bake skipped: UV rasterisation covered nothing")
        return None, None, {}

    pos = pos[covered]
    nrm = torch.nn.functional.normalize(nrm[covered], dim=-1, eps=1e-12)
    tan = torch.nn.functional.normalize(tan[covered], dim=-1, eps=1e-12)
    tan = torch.nn.functional.normalize(
        tan - nrm * (nrm * tan).sum(-1, keepdim=True), dim=-1, eps=1e-12)
    bit = torch.cross(nrm, tan, dim=-1)

    # --- 2. shoot at the dense mesh ----------------------------------------
    bvh = cumesh.cuBVH(high_vertices.contiguous(), high_faces.contiguous().int())
    origins = pos + nrm * cage_out
    hit_pos, face_id, depth = bvh.ray_trace(origins.contiguous(), (-nrm).contiguous())

    valid = (face_id >= 0) & (depth < cage_out + ray_len)
    # Concave areas are better reached from the other side.
    missed = ~valid
    if bool(missed.any()):
        o2 = pos[missed] - nrm[missed] * cage_out
        h2, f2, d2 = bvh.ray_trace(o2.contiguous(), nrm[missed].contiguous())
        ok2 = (f2 >= 0) & (d2 < cage_out + ray_len)
        idx = torch.nonzero(missed, as_tuple=True)[0][ok2]
        hit_pos[idx] = h2[ok2]
        face_id[idx] = f2[ok2]
        valid[idx] = True

    # --- 3. smooth dense normal at the hit, expressed in the low frame ------
    safe_face = face_id.clamp(min=0).long()
    tri = high_faces.long()[safe_face]
    a, b, c = (high_vertices[tri[:, 0]], high_vertices[tri[:, 1]],
               high_vertices[tri[:, 2]])
    bary = _barycentric(hit_pos, a, b, c).clamp(0.0, 1.0)
    n_hit = (high_n[tri[:, 0]] * bary[:, 0:1]
             + high_n[tri[:, 1]] * bary[:, 1:2]
             + high_n[tri[:, 2]] * bary[:, 2:3])
    n_hit = torch.nn.functional.normalize(n_hit, dim=-1, eps=1e-12)

    # Reject hits on a surface facing away from us: the ray went through the
    # shell. Keeping them is what turns a normal map into confetti.
    agreement = (n_hit * nrm).sum(-1)
    pierced = valid & (agreement < _MIN_NORMAL_AGREEMENT)
    valid = valid & (agreement >= _MIN_NORMAL_AGREEMENT)

    ts = torch.stack([(n_hit * tan).sum(-1),
                      (n_hit * bit).sum(-1),
                      (n_hit * nrm).sum(-1)], dim=-1)
    ts = torch.nn.functional.normalize(ts, dim=-1, eps=1e-12)
    # Texels that found nothing keep a flat normal rather than noise.
    flat = torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(ts)
    ts = torch.where(valid[:, None], ts, flat)
    pierced_ratio = float(pierced.float().mean())

    # --- 4. write the image ------------------------------------------------
    encoded = ((ts * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    image = torch.zeros((size, size, 3), dtype=torch.uint8, device=dev)
    image[..., 2] = 255  # flat normal everywhere the mesh is not
    image[covered] = encoded
    image[..., 0][~covered] = 128
    image[..., 1][~covered] = 128

    rgb = image.cpu().numpy()
    mask = covered.cpu().numpy()
    # No flip here any more: uvraster.atlas already hands the buffer back
    # top-down, which is what every writer downstream expects.
    rgb = _fill_gutters(rgb, mask)

    # --- 5. carry the dense mesh's textures into the new UV layout ---------
    transferred: dict = {}
    if transfer and high_uv is not None:
        high_uv = high_uv.to(dev, torch.float32)
        if high_uv.shape[0] == high_vertices.shape[0]:
            uv_hit = (high_uv[tri[:, 0]] * bary[:, 0:1]
                      + high_uv[tri[:, 1]] * bary[:, 1:2]
                      + high_uv[tri[:, 2]] * bary[:, 2:3])
            # Texels whose ray found nothing would sample a random spot; hold
            # them at the island edge instead so dilation fills them sensibly.
            uv_hit = uv_hit.clamp(0.0, 1.0)

            # Texels whose ray found nothing, or punched through the shell,
            # carry a meaningless face index and would sample an arbitrary spot
            # in the source atlas. Left in, those few per mille show up as
            # scattered wrong-coloured specks all over the model - visible, and
            # impossible to attribute without comparing against an unbudgeted
            # generation. They are marked unfilled instead, so the gutter fill
            # gives them their nearest real neighbour's colour.
            trustworthy = torch.zeros_like(covered)
            trustworthy[covered] = valid
            trust_np = trustworthy.cpu().numpy()

            for name, array in transfer.items():
                sampled = _sample(array, uv_hit, dev)
                if sampled is None:
                    continue
                channels = sampled.shape[-1]
                img = torch.zeros((size, size, channels), dtype=torch.uint8, device=dev)
                img[covered] = sampled
                out = _fill_gutters(img.cpu().numpy(), trust_np)
                transferred[name] = out
            log.info("transferred %d map(s) into the new UV layout: %s",
                     len(transferred), ", ".join(transferred))
        else:
            log.warning("cannot transfer textures: %d UVs for %d dense vertices",
                        high_uv.shape[0], high_vertices.shape[0])

    coverage = float(valid.float().mean())
    report = BakeReport(size=size, coverage=coverage,
                        faces_high=int(high_faces.shape[0]),
                        faces_low=int(low_faces.shape[0]),
                        pierced=pierced_ratio)
    log.info("normal bake: %dx%d, %.1f%% usable hits, %.1f%% rejected as "
             "through-shell", size, size, coverage * 100, pierced_ratio * 100)
    return rgb, report, transferred

