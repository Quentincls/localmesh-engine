"""Draw triangles into a UV atlas, in PyTorch, without nvdiffrast.

nvdiffrast is excellent and its licence forbids selling anything built with it
(Nvidia Source Code License, clause 3.3: research or evaluation only). It is
used at eight points in LumenGen, and six of them ask for the same thing: given
a mesh with UV coordinates, tell every texel of the atlas which point of the
model it belongs to.

That is a far smaller problem than general rendering, and this module solves
only it. In UV space the triangles are already flat inside the unit square:
there is no camera, no perspective divide, no depth test - islands do not
overlap by construction, since that is what an atlas *is*. What remains is the
oldest operation in graphics, filling a triangle by barycentric coordinates.

Conventions are copied from nvdiffrast rather than chosen, so the callers can be
switched over without touching anything downstream: row zero is the bottom of
the image, a texel is sampled at its centre, and a point exactly on an edge
belongs to the triangle.
"""
from __future__ import annotations

import logging

log = logging.getLogger("localmesh_engine.uvraster")

#: Texels a batch may cover at once. A batch allocates a grid the size of its
#: largest triangle repeated over the whole batch, so the count of triangles is
#: the wrong thing to fix: 4096 sub-pixel triangles fill 4096 texels, and 4096
#: screen-wide ones fill sixteen billion. Fixing the *area* instead lets a
#: dense mesh at low resolution go through in a handful of passes rather than
#: sixteen thousand - measured at 60 s against 4 s on a 300k-face silhouette.
#: Around ten intermediate tensors live at once, so this is ~600 MB.
_BUDGET = 16 << 20


def rasterise(uv, faces, size: int, attributes, depth=None, nearest="max",
              inv_w=None):
    """Fill the atlas and interpolate `attributes` across every covered texel.

    `uv` is (N, 2) in [0, 1], `faces` is (F, 3), and `attributes` is a list of
    (N, C) tensors carried on the vertices - a position, a normal, anything.

    With `depth` - one value per vertex - overlapping triangles are resolved by
    a z-buffer instead of by whoever writes last, and only the winner's
    attributes survive. That is what an atlas never needs, since its islands do
    not overlap, and what a view of a solid object always does.

    With `inv_w`, the interpolation is perspective-correct. Barycentric weights
    computed on screen are linear in screen space, which is what an attribute
    is only when the surface is parallel to it - true of a UV atlas, false of
    anything seen through a camera. A floor tiled in a texture is the standard
    demonstration: interpolated flat, its squares stay the same size all the
    way to the horizon. The remedy is as old as the problem - interpolate the
    attribute over w and 1/w separately, then divide.

    Returns the interpolated attributes, each (size, size, C), and a boolean
    coverage mask, in nvdiffrast's bottom-up order.
    """
    import torch

    dev = uv.device
    uv = uv.to(torch.float32)
    faces = faces.to(torch.int64)

    # Straight to texel coordinates. A texel's centre is at (col + 0.5), which
    # is where nvdiffrast samples, so the +0.5 has to be here and not later.
    pixels = uv * float(size)
    corners = pixels[faces]                                   # (F, 3, 2)

    lo = corners.amin(dim=1)
    hi = corners.amax(dim=1)
    x0 = lo[:, 0].floor().clamp(0, size - 1).to(torch.int64)
    y0 = lo[:, 1].floor().clamp(0, size - 1).to(torch.int64)
    x1 = hi[:, 0].ceil().clamp(0, size - 1).to(torch.int64)
    y1 = hi[:, 1].ceil().clamp(0, size - 1).to(torch.int64)
    span_x = (x1 - x0 + 1).clamp(min=1)
    span_y = (y1 - y0 + 1).clamp(min=1)

    # Twice the signed area. A degenerate triangle has none and is skipped
    # rather than dividing by zero.
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
         - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    alive = area.abs() > 1e-12

    total = size * size
    channels = [t.shape[1] for t in attributes]
    carried = list(attributes)
    if inv_w is not None:
        inv_w = inv_w if inv_w.ndim == 2 else inv_w[:, None]
        carried = [t * inv_w for t in carried] + [inv_w]
    if depth is not None:
        # The depth rides along as one more interpolated channel, so a fragment
        # already knows how far away it is by the time the z-test runs.
        carried = carried + [depth if depth.ndim == 2 else depth[:, None]]
    if carried:
        packed = torch.cat([t.to(torch.float32) for t in carried], dim=1)
    else:
        # Coverage alone - a silhouette. Nothing is interpolated, but the
        # machinery below wants a value per vertex, so it gets an empty one.
        packed = torch.zeros(uv.shape[0], 0, device=dev, dtype=torch.float32)
    out = torch.zeros(total, packed.shape[1], device=dev, dtype=torch.float32)
    covered = torch.zeros(total, dtype=torch.bool, device=dev)
    zbuf = None
    if depth is not None:
        far = float("-inf") if nearest == "max" else float("inf")
        zbuf = torch.full((total,), far, device=dev, dtype=torch.float32)

    index = torch.nonzero(alive, as_tuple=False).squeeze(1)
    # Sorted by footprint so each batch is homogeneous: a batch's grid is as
    # large as its largest triangle, and mixing a screen-wide one with a
    # thousand two-texel ones would allocate that grid a thousand times.
    order = torch.argsort(span_x[index] * span_y[index])
    index = index[order]

    # With a z-buffer it takes two passes over the triangles: the first only
    # records how near the nearest fragment is at each texel, the second writes
    # the attributes of whoever matches. Resolving as we go would need the
    # fragments held in memory all at once, which on a dense mesh is worse than
    # rasterising twice.
    passes = ("depth", "write") if zbuf is not None else ("write",)
    for phase in passes:
        start = 0
        while start < len(index):
            # The list is sorted by footprint, so the triangle at `start` is
            # the smallest of what is left: it gives the optimistic batch size.
            # Then shrink until the batch's *largest* member also fits, which
            # takes a step or two because the sort keeps a batch homogeneous.
            small = int(span_x[index[start]].item() * span_y[index[start]].item())
            n = max(1, min(len(index) - start, _BUDGET // max(small, 1)))
            while True:
                chunk = index[start:start + n]
                width = int(span_x[chunk].max().item())
                height = int(span_y[chunk].max().item())
                if n == 1 or n * width * height <= _BUDGET:
                    break
                n = max(1, int(n * _BUDGET / (n * width * height)))
            _fill(chunk, corners, area, x0, y0, width, height,
                  faces, packed, out, covered, size, dev, zbuf, phase, nearest)
            start += n

    if inv_w is not None:
        # Undo the premultiplication. Uncovered texels hold zero over zero, so
        # the denominator is floored - they are masked off by `covered` anyway.
        weight = out[:, sum(channels):sum(channels) + 1]
        out = out[:, :sum(channels)] / weight.clamp(min=1e-20)

    pieces, at = [], 0
    for width in channels:
        pieces.append(out[:, at:at + width].reshape(size, size, width))
        at += width
    return pieces, covered.reshape(size, size)


def atlas(uv, faces, size: int, attributes=(), as_numpy: bool = True):
    """The whole idiom the callers use, in one call, the right way up.

    `rasterise` works in nvdiffrast's bottom-up order to stay comparable with
    it; everything downstream in LumenGen thinks top-down and used to flip by
    hand at each of the six call sites. Doing it here removes the one mistake
    that keeps happening - a texture that comes out upside down looks so wrong
    that it is caught immediately, but a *normal map* flipped the same way only
    shows as lighting that is subtly inside out.
    """
    import torch

    dev = uv.device
    prepared = [t if t.ndim == 2 else t[:, None] for t in attributes]
    values, covered = rasterise(uv, faces.to(torch.int64), size, prepared)
    covered = torch.flip(covered, dims=[0])
    values = [torch.flip(v, dims=[0]) for v in values]
    if not as_numpy:
        return values, covered
    import numpy as np

    return ([np.ascontiguousarray(v.cpu().numpy()) for v in values],
            np.ascontiguousarray(covered.cpu().numpy()))


def screen(clip_xy, faces, size: int, attributes=(), depth=None,
           nearest: str = "max", as_numpy: bool = True):
    """The same rasteriser, fed clip coordinates instead of texture ones.

    A view of a mesh and a texture atlas differ in two things only: the square
    runs from -1 to 1 rather than 0 to 1, and a view has depth - two triangles
    can land on the same pixel and one of them is in front. Everything else,
    including the fill rule, is shared.

    `nearest` says which end of `depth` is closer to the camera. LumenGen's
    orthographic frames put the camera down +z, so larger is nearer.

    Unlike `atlas`, this returns bottom-up, exactly as nvdiffrast does. The
    callers here are views, and a view's orientation is already threaded
    through the pose search, the silhouette comparison and the photo
    projection; turning it over at the source would flip all three at once.
    """
    import numpy as np
    import torch

    uv = (clip_xy[:, :2] + 1.0) * 0.5
    prepared = [t if t.ndim == 2 else t[:, None] for t in attributes]
    values, covered = rasterise(uv, faces.to(torch.int64), size, prepared,
                                depth=depth, nearest=nearest)
    if not as_numpy:
        return values, covered
    return ([np.ascontiguousarray(v.cpu().numpy()) for v in values],
            np.ascontiguousarray(covered.cpu().numpy()))


def camera(clip, faces, size: int, attributes=()):
    """Render a mesh from a camera: `clip` is (N, 4), homogeneous, undivided.

    This is `screen` with the perspective put back in - the divide by w, the
    depth test on the divided z, and perspective-correct attributes. Vertices
    behind the eye are dropped rather than clipped: a proper near-plane clip
    would split triangles, and LumenGen only ever frames a whole object from
    outside it, so nothing that matters is ever half in front of the eye.

    Returns the attributes, (size, size, C) each, top-down, plus coverage.
    """
    import torch

    w = clip[:, 3:4]
    behind = w <= 1e-6
    ndc = clip[:, :3] / w.clamp(min=1e-6)
    # A vertex behind the eye is pushed far outside the frame instead, so any
    # triangle touching it falls off the edge rather than folding back into it.
    ndc = torch.where(behind, torch.full_like(ndc, 1e4), ndc)

    prepared = [t if t.ndim == 2 else t[:, None] for t in attributes]
    values, covered = rasterise((ndc[:, :2] + 1.0) * 0.5, faces.to(torch.int64),
                                size, prepared, depth=ndc[:, 2:3],
                                nearest="min", inv_w=1.0 / w.clamp(min=1e-6))
    covered = torch.flip(covered, dims=[0])
    values = [torch.flip(v, dims=[0]) for v in values]
    return values, covered


def sample(texture, st):
    """Bilinear texture lookup, the way nvdiffrast's `texture` does it.

    `texture` is (H, W, C) with row zero at the top, `st` is (..., 2) in [0, 1]
    with v already flipped by the caller, as glTF wants.
    """
    import torch

    grid = (st * 2.0 - 1.0).reshape(1, -1, 1, 2)
    plane = texture.permute(2, 0, 1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(plane, grid, mode="bilinear",
                                          padding_mode="border",
                                          align_corners=False)
    return out.squeeze(0).squeeze(-1).T.reshape(*st.shape[:-1], texture.shape[2])


def _fill(chunk, corners, area, x0, y0, width, height,
          faces, packed, out, covered, size, dev,
          zbuf=None, phase="write", nearest="max"):
    """Fill one batch of triangles into the shared buffers."""
    import torch

    ys = torch.arange(height, device=dev)
    xs = torch.arange(width, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    px = (x0[chunk][:, None, None] + gx[None]).to(torch.float32) + 0.5
    py = (y0[chunk][:, None, None] + gy[None]).to(torch.float32) + 0.5

    a = corners[chunk, 0][:, None, None, :]
    b = corners[chunk, 1][:, None, None, :]
    c = corners[chunk, 2][:, None, None, :]

    # Everything measured from the triangle's own first corner. Written the
    # obvious way - cross products of absolute pixel coordinates - the terms
    # reach 2048 while the triangle itself spans two or three texels, and
    # float32 loses the difference: the edges disagreed with nvdiffrast on
    # about 0.2 % of texels, in both directions, which is the signature of
    # precision rather than of a different fill rule. Subtracting the corner
    # first keeps every quantity triangle-sized.
    e1 = b - a
    e2 = c - a
    dx = px - a[..., 0]
    dy = py - a[..., 1]

    two_area = area[chunk][:, None, None]
    # Barycentric weights, from the signed area. Dividing by the signed value
    # rather than its magnitude is what handles both windings without a case.
    w1 = (dx * e2[..., 1] - dy * e2[..., 0]) / two_area
    w2 = (e1[..., 0] * dy - e1[..., 1] * dx) / two_area
    w0 = 1.0 - w1 - w2

    # The top-left rule. A texel whose centre falls exactly on an edge belongs
    # to one of the two triangles sharing it, never both, and the convention
    # decides which: the edge is claimed only if it is a "top" or "left" one.
    #
    # Without it a pixel on any boundary is claimed by everything touching it.
    # That is invisible on a lone triangle - three synthetic cases out of five
    # matched to the texel - and shows up the moment vertices land on texel
    # centres, which is constantly, because xatlas packs islands on grid-
    # aligned rectangles. It was the whole of the 0.2 % disagreement.
    flip = torch.where(two_area < 0, -1.0, 1.0)
    edges = ((c - b)[..., :2] * flip[..., None],
             (a - c)[..., :2] * flip[..., None],
             (b - a)[..., :2] * flip[..., None])
    inside = None
    for w, e in zip((w0, w1, w2), edges):
        ex, ey = e[..., 0], e[..., 1]
        claims = (ey < 0) | ((ey == 0) & (ex > 0))
        ok = torch.where(claims, w >= 0, w > 0)
        inside = ok if inside is None else (inside & ok)
    ix = (x0[chunk][:, None, None] + gx[None])
    iy = (y0[chunk][:, None, None] + gy[None])
    inside &= (ix < size) & (iy < size)
    if not bool(inside.any()):
        return

    tri, row, col = torch.nonzero(inside, as_tuple=True)
    flat = iy[tri, row, col] * size + ix[tri, row, col]
    verts = faces[chunk][tri]
    weights = torch.stack([w0[tri, row, col], w1[tri, row, col],
                           w2[tri, row, col]], dim=-1)
    value = (packed[verts] * weights[..., None]).sum(dim=1)

    if zbuf is None:
        # Islands do not overlap, so last writer wins is as good as any rule -
        # and it matches what a rasteriser with a constant depth does.
        out[flat] = value
        covered[flat] = True
        return

    z = value[:, -1]
    if phase == "depth":
        zbuf.scatter_reduce_(0, flat, z,
                             reduce="amax" if nearest == "max" else "amin",
                             include_self=True)
        return

    # Only the fragment the depth pass elected. The tolerance is there because
    # the two passes recompute the same interpolation and floating point does
    # not promise the same bits twice.
    wins = (z - zbuf[flat]).abs() <= 1e-6 * (1.0 + z.abs())
    if not bool(wins.any()):
        return
    out[flat[wins]] = value[wins]
    covered[flat[wins]] = True
