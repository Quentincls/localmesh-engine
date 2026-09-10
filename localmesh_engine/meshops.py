"""Artist-facing mesh finishing, on top of CuMesh's CUDA kernels.

Everything here runs on the GPU. The vocabulary is deliberately the artist's
(poly budget, topology, floaters, UVs) rather than the library's.

Order matters and is not negotiable:
    repair -> floaters -> holes -> retopology -> poly budget -> UVs
Unwrapping before decimation would throw the UVs away; decimating before
removing floaters spends budget on garbage.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import cumesh
import torch

log = logging.getLogger("localmesh_engine.meshops")

Topology = Literal["tri", "quad"]

# CuMesh.simplify never lands exactly on target; accept this much overshoot
# before spending another (expensive) quad remesh pass.
_BUDGET_TOLERANCE = 0.12
_MAX_QUAD_PASSES = 3


@dataclass
class FinishSettings:
    """The artist panel, one-to-one."""

    # Poly budget, in faces of the final mesh. None = keep whatever comes out.
    target_faces: Optional[int] = None
    topology: Topology = "tri"

    # Cleanup
    repair: bool = True
    #: Vertices closer than this fraction of the bounding-box diagonal are
    #: merged before anything else touches the mesh. Mostly this reunites the
    #: two sides of every UV seam, which is what stops decimation from tearing
    #: the surface open.
    weld_tolerance: float = 1e-4
    remove_floaters: bool = True
    #: A component counts as debris below this fraction of the LARGEST
    #: component's area. Measuring against the largest piece rather than the
    #: total is what makes this safe on assets built from many small parts.
    floater_area_ratio: float = 0.001
    #: Hard ceiling on what floater removal may delete, as a fraction of total
    #: surface. If the filter wants more than this, it has misunderstood the
    #: model and is skipped entirely.
    max_floater_removal_ratio: float = 0.05
    #: A small component only counts as debris if it also sits this far away
    #: from the body, as a fraction of the bounding-box diagonal. Voxel-derived
    #: meshes are full of small patches lying flush against the main shell;
    #: they look like one surface and deleting them punches visible holes.
    floater_min_distance: float = 0.004
    fill_holes: bool = True
    #: max perimeter of a hole to close, relative to the model's largest dimension
    hole_perimeter_ratio: float = 0.03

    # Retopology (only used when topology == "quad" or force_remesh)
    force_remesh: bool = False
    remesh_resolution: Optional[int] = None  # None = derived from poly budget
    remesh_band: float = 1.0
    #: 0..1, how hard vertices are snapped back onto the original surface
    remesh_project: float = 0.9
    remove_inner_faces: bool = True

    # UVs
    unwrap_uv: bool = True
    #: larger angle = fewer, bigger charts = fewer seams but more distortion
    chart_cone_half_angle_rad: float = math.pi / 2
    chart_refine_iterations: int = 0
    chart_global_iterations: int = 1
    chart_smooth_strength: float = 1.0


@dataclass
class FinishReport:
    """What actually happened - surfaced in the UI, not just logged."""

    faces_in: int = 0
    faces_out: int = 0
    verts_in: int = 0
    verts_out: int = 0
    topology: Topology = "tri"
    #: Duplicate vertices merged before anything else - almost all of them the
    #: two sides of a UV seam.
    welded: int = 0
    floaters_removed: int = 0
    holes_filled: bool = False
    remesh_resolution: Optional[int] = None
    quad_passes: int = 0
    uv_charts: Optional[int] = None
    budget_hit: Optional[bool] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "faces_in": self.faces_in,
            "faces_out": self.faces_out,
            "verts_in": self.verts_in,
            "verts_out": self.verts_out,
            "topology": self.topology,
            "welded": self.welded,
            "floaters_removed": self.floaters_removed,
            "holes_filled": self.holes_filled,
            "remesh_resolution": self.remesh_resolution,
            "quad_passes": self.quad_passes,
            "uv_charts": self.uv_charts,
            "budget_hit": self.budget_hit,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _aabb(vertices: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Return (center, largest extent) of the bounding box."""
    lo = vertices.min(dim=0).values
    hi = vertices.max(dim=0).values
    return (lo + hi) * 0.5, float((hi - lo).max().item())


def _surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> float:
    v = vertices[faces.long()]
    cross = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)
    return float(0.5 * cross.norm(dim=-1).sum().item())


def _weld(vertices: torch.Tensor, faces: torch.Tensor,
          tolerance: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge vertices that occupy the same position.

    A textured mesh out of TRELLIS is *split along every UV seam*: the two
    sides of a seam share a position but are separate vertices. That is normal
    and invisible at full density - and catastrophic as soon as you decimate,
    because each side is simplified independently, drifts away from its twin,
    and the seam tears into a real hole. On a generated eye that turned into
    thousands of gaps scattered over the whole model.

    Welding on position alone is safe here precisely because the UVs are about
    to be rebuilt anyway.
    """
    if tolerance <= 0:
        return vertices, faces

    keys = torch.round(vertices / tolerance).to(torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = int(inverse.max()) + 1

    # Average the positions merged into each slot rather than snapping to the
    # quantisation grid, so welding does not itself shift the surface.
    merged = torch.zeros((count, 3), dtype=vertices.dtype, device=vertices.device)
    merged.index_add_(0, inverse, vertices)
    hits = torch.zeros(count, dtype=vertices.dtype, device=vertices.device)
    hits.index_add_(0, inverse, torch.ones_like(inverse, dtype=vertices.dtype))
    merged = merged / hits.clamp(min=1).unsqueeze(-1)

    new_faces = inverse[faces.long()]
    # Faces whose corners collapsed onto each other are no longer triangles.
    ok = ((new_faces[:, 0] != new_faces[:, 1])
          & (new_faces[:, 1] != new_faces[:, 2])
          & (new_faces[:, 0] != new_faces[:, 2]))
    return merged.contiguous(), new_faces[ok].to(torch.int32).contiguous()


def _handle(vertices: torch.Tensor, faces: torch.Tensor) -> cumesh.CuMesh:
    m = cumesh.CuMesh()
    m.init(vertices.contiguous().float(), faces.contiguous().int())
    return m


#: Measured on the reference sphere: a narrow-band DC pass at resolution R emits
#: ~8.4*R^2 quads, and the kernel triangulates its own output, so ~16.8*R^2
#: triangles. Only a starting guess - the real constant moves with surface area
#: and genus, which is why _retopologise then corrects from the measurement.
_DC_FACES_PER_RES2 = 16.8

_RES_MIN, _RES_MAX = 32, 1024


def _resolution_for_budget(target_faces: int) -> int:
    """First-guess dual-contouring resolution for a face budget."""
    res = math.sqrt(max(target_faces, 1) / _DC_FACES_PER_RES2)
    res = int(round(res / 8.0) * 8)
    return max(_RES_MIN, min(res, _RES_MAX))


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def finish(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    settings: FinishSettings,
    progress=None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], FinishReport]:
    """Run the artist finishing pipeline.

    Returns (vertices, faces, uvs_or_None, report). All tensors stay on the GPU.
    `progress` is an optional callable(stage: str, fraction: float).
    """
    if vertices.device.type != "cuda":
        vertices = vertices.cuda()
    if faces.device.type != "cuda":
        faces = faces.cuda()

    rep = FinishReport(
        faces_in=int(faces.shape[0]),
        verts_in=int(vertices.shape[0]),
        topology=settings.topology,
    )

    def step(stage: str, frac: float):
        log.info("finish: %s", stage)
        if progress:
            progress(stage, frac)

    # -- 1. repair -----------------------------------------------------------
    if settings.repair:
        step("repairing geometry", 0.05)

        # Must come first: everything downstream assumes a stitched surface.
        _, extent = _aabb(vertices)
        before_v = int(vertices.shape[0])
        vertices, faces = _weld(vertices, faces, settings.weld_tolerance * extent)
        rep.welded = before_v - int(vertices.shape[0])

        m = _handle(vertices, faces)
        m.remove_degenerate_faces()
        m.remove_duplicate_faces()
        m.repair_non_manifold_edges()
        m.unify_face_orientations()
        m.remove_unreferenced_vertices()
        vertices, faces = m.read()

    # -- 2. floaters ---------------------------------------------------------
    if settings.remove_floaters:
        step("removing floating parts", 0.12)
        vertices, faces = _remove_floaters(vertices, faces, settings, rep)

    # -- 3. holes ------------------------------------------------------------
    if settings.fill_holes:
        step("closing holes", 0.2)
        _, extent = _aabb(vertices)
        m = _handle(vertices, faces)
        m.fill_holes(settings.hole_perimeter_ratio * extent)
        vertices, faces = m.read()
        rep.holes_filled = True

    # -- 4. retopology / poly budget ----------------------------------------
    want_quad = settings.topology == "quad"
    if want_quad or settings.force_remesh:
        vertices, faces = _retopologise(vertices, faces, settings, rep, step)
    elif settings.target_faces:
        # Le nombre est ecrit a l anglaise (« 150,000 ») et le libelle sort
        # des catalogues des qu il porte un chiffre. L etape se dit sans.
        step("decimating", 0.55)
        m = _handle(vertices, faces)
        m.simplify(int(settings.target_faces))
        m.remove_unreferenced_vertices()
        vertices, faces = m.read()

    if settings.target_faces:
        err = abs(int(faces.shape[0]) - settings.target_faces) / settings.target_faces
        rep.budget_hit = err <= _BUDGET_TOLERANCE
        if not rep.budget_hit:
            rep.notes.append(
                f"poly budget missed: asked {settings.target_faces:,}, "
                f"got {int(faces.shape[0]):,}"
            )

    # -- 5. UVs --------------------------------------------------------------
    uvs = None
    if settings.unwrap_uv:
        step("unwrapping UVs", 0.8)
        m = _handle(vertices, faces)
        out = m.uv_unwrap(
            compute_charts_kwargs=dict(
                threshold_cone_half_angle_rad=settings.chart_cone_half_angle_rad,
                refine_iterations=settings.chart_refine_iterations,
                global_iterations=settings.chart_global_iterations,
                smooth_strength=settings.chart_smooth_strength,
            )
        )
        # (vertices, faces, uvs, [vmaps])
        vertices, faces, uvs = out[0], out[1], out[2]

    rep.faces_out = int(faces.shape[0])
    rep.verts_out = int(vertices.shape[0])
    step("done", 1.0)
    return vertices, faces, uvs, rep


def _remove_floaters(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    settings: FinishSettings,
    rep: FinishReport,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delete detached specks without deleting the model.

    Generated meshes routinely carry a few stray voxel islands, and they are
    ugly. But an ornate asset - filigree, chain, foliage - is *legitimately*
    made of dozens of small disconnected shells, and a threshold expressed
    against total surface area wipes all of them out. That failure is silent:
    the job still reports success and the file still opens.

    So debris is defined relative to the largest component, and the whole step
    is bounded: if it wants to remove more than `max_floater_removal_ratio` of
    the surface, it stands down and says so.
    """
    m = _handle(vertices, faces)
    m.get_connected_components()
    count, labels = m.read_connected_components()
    if count <= 1:
        return vertices, faces

    tri = vertices[faces.long()]
    face_area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0],
                                  tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    comp_area = torch.zeros(count, device=vertices.device, dtype=face_area.dtype)
    comp_area.index_add_(0, labels.long(), face_area)

    total = float(comp_area.sum())
    largest = float(comp_area.max())
    if total <= 0 or largest <= 0:
        return vertices, faces

    small = comp_area < settings.floater_area_ratio * largest
    if not bool(small.any()):
        return vertices, faces

    # Being small is not enough. A mesh reconstructed from voxels is riddled
    # with little patches that sit *flush against* the main shell: visually
    # they are the surface. Deleting them leaves a scatter of holes across the
    # model - which is exactly what happened to a generated eye, while the
    # unfinished version of the same asset was spotless.
    #
    # So a component must also be spatially detached from the body.
    body = int(comp_area.argmax())
    body_faces = faces[labels.long() == body]
    detached = torch.zeros_like(small)
    if body_faces.shape[0] > 0:
        bvh = cumesh.cuBVH(vertices.contiguous(), body_faces.contiguous().int())
        _, extent = _aabb(vertices)
        threshold = settings.floater_min_distance * extent

        for comp in torch.nonzero(small, as_tuple=True)[0].tolist():
            if comp == body:
                continue
            verts = vertices[faces[labels.long() == comp].long().reshape(-1)]
            if verts.shape[0] == 0:
                continue
            # A handful of samples is plenty to tell "lying on it" from "far".
            if verts.shape[0] > 512:
                verts = verts[torch.randperm(verts.shape[0], device=verts.device)[:512]]
            dist = bvh.unsigned_distance(verts.contiguous())
            if isinstance(dist, (tuple, list)):
                dist = dist[0]
            if float(dist.median()) > threshold:
                detached[comp] = True

    doomed = small & detached
    if not bool(doomed.any()):
        kept = int(small.sum())
        rep.notes.append(
            f"{kept} petits morceaux conservés : ils touchent la surface "
            f"principale, ce sont des fragments du modèle et non des débris."
        )
        return vertices, faces

    removed_ratio = float(comp_area[doomed].sum()) / total
    if removed_ratio > settings.max_floater_removal_ratio:
        rep.notes.append(
            f"Floater cleanup skipped: it would have removed {removed_ratio:.0%} "
            f"of the surface across {int(doomed.sum())} parts. This model is made "
            f"of many small pieces, so they were kept."
        )
        return vertices, faces

    keep = ~doomed[labels.long()]
    before = int(faces.shape[0])
    kept_faces = faces[keep].contiguous()
    if kept_faces.shape[0] == 0:
        return vertices, faces

    m = _handle(vertices, kept_faces)
    m.remove_unreferenced_vertices()
    vertices, faces = m.read()

    rep.floaters_removed = before - int(faces.shape[0])
    if rep.floaters_removed:
        rep.notes.append(
            f"{rep.floaters_removed:,} faces of detached debris removed "
            f"({int(doomed.sum())} parts, {removed_ratio:.1%} of the surface)"
        )
    return vertices, faces


def _retopologise(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    settings: FinishSettings,
    rep: FinishReport,
    step,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rebuild topology with dual contouring, converging on the poly budget.

    Quads cannot be decimated afterwards without destroying the quad structure,
    so for quad output the budget is hit by choosing the grid resolution. The
    face count scales with resolution squared, which makes the correction step
    a simple sqrt - two passes are usually enough.
    """
    quad = settings.topology == "quad"
    fn = (cumesh.remeshing.remesh_narrow_band_dc_quad if quad
          else cumesh.remeshing.remesh_narrow_band_dc)

    if settings.remesh_resolution:
        res = int(settings.remesh_resolution)
    elif settings.target_faces:
        res = _resolution_for_budget(settings.target_faces)
    else:
        res = 256

    best: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    best_err = float("inf")

    passes = 1 if not (quad and settings.target_faces) else _MAX_QUAD_PASSES
    for i in range(passes):
        # Deux libellés FIXES plutôt qu'une f-string : `nomEtape`
        # (Planche.tsx) traduit par correspondance exacte, et « rebuilding
        # quad topology @ 512 » traversait les sept langues en anglais brut.
        # La résolution vit dans le journal, pas à l'écran.
        log.info("rebuilding %s topology @ %d",
                 "quad" if quad else "triangle", res)
        step("rebuilding quad topology" if quad
             else "rebuilding triangle topology",
             0.3 + 0.2 * i / max(passes, 1))
        center, extent = _aabb(vertices)
        scale = (res + 3 * settings.remesh_band) / res * extent

        out = fn(
            vertices, faces,
            center=center,
            scale=scale,
            resolution=res,
            band=settings.remesh_band,
            project_back=settings.remesh_project,
            remove_inner_faces=settings.remove_inner_faces,
            verbose=False,
        )
        nv, nf = out[0], out[1]
        rep.quad_passes = i + 1
        rep.remesh_resolution = res

        if not settings.target_faces:
            return nv, nf

        got = int(nf.shape[0])
        err = abs(got - settings.target_faces) / settings.target_faces
        if err < best_err:
            best, best_err = (nv, nf), err
        if err <= _BUDGET_TOLERANCE or i == passes - 1:
            break

        # face count ~ res^2  ->  correct in sqrt space, damped to avoid ringing
        ratio = math.sqrt(settings.target_faces / max(got, 1))
        ratio = max(0.4, min(ratio, 2.5))
        new_res = int(round(res * ratio / 8.0) * 8)
        new_res = max(_RES_MIN, min(new_res, _RES_MAX))
        if new_res == res:
            break
        res = new_res

    nv, nf = best  # type: ignore[misc]

    # The DC kernel triangulates its own output (quads exist only internally, and
    # glTF cannot carry them anyway), so decimating afterwards costs no topology
    # we still have. Below roughly res 32 the grid degenerates, which puts a hard
    # floor on how few faces retopology alone can reach - decimation covers the
    # rest of the way down to the artist's budget.
    if settings.target_faces and int(nf.shape[0]) > settings.target_faces * (1 + _BUDGET_TOLERANCE):
        step("decimating", 0.55)
        if quad:
            rep.notes.append(
                f"retopology floors at ~{int(nf.shape[0]):,} faces for this shape "
                f"(grid {res}); decimated the rest of the way"
            )
        m = _handle(nv, nf)
        m.simplify(int(settings.target_faces))
        m.remove_unreferenced_vertices()
        nv, nf = m.read()

    return nv, nf
