"""Offscreen turntable renders of a finished asset, on the GPU.

Used for library thumbnails in the app, and as the visual QA gate during
development: an asset that exports cleanly can still be wrong, and the only way
to know is to look at it.

Deliberately simple shading - base colour, a lambert key, a hemispheric fill and
a rim - because the point is to *read the silhouette and the albedo*, not to
match the in-app three.js viewport.

Rasterised by `uvraster`, in PyTorch. The shading below is unchanged from the
nvdiffrast version it replaced, deliberately: a thumbnail is what the user
recognises an asset by in the library, and there was no reason for the whole
library to change appearance because of a licence.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


#: How much larger than the final frame each view is rendered, for antialiasing.
_SUPERSAMPLE = 3


def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float32)
    s = np.cross(f, u)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _perspective(fov_deg, aspect, near=0.01, far=100.0):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _collect(scene):
    """Flatten a GLB (trimesh Scene or Trimesh) into one buffer set."""
    import trimesh

    geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
    verts, faces, uvs, colors = [], [], [], []
    textures, mrs, aos = [], [], []
    offset = 0
    for g in geoms:
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        verts.append(np.asarray(g.vertices, dtype=np.float32))
        faces.append(np.asarray(g.faces, dtype=np.int32) + offset)

        uv = getattr(getattr(g, "visual", None), "uv", None)
        uvs.append(np.asarray(uv, dtype=np.float32) if uv is not None
                   else np.zeros((len(g.vertices), 2), dtype=np.float32))

        mat = getattr(getattr(g, "visual", None), "material", None)
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        textures.append(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                        if img is not None else None)

        mr_img = getattr(mat, "metallicRoughnessTexture", None)
        mrs.append(np.asarray(mr_img.convert("RGB"), dtype=np.float32) / 255.0
                   if mr_img is not None else None)

        # L'occlusion ambiante est RETIREE de l'albedo a la cuisson (sinon le
        # renderer la doublerait), et personne ne la remettait ici : toutes
        # les vignettes sortaient plus claires et plus plates que ce que le
        # materiau decrit. Mesure sur la statue de bronze : albedo 1,77 fois
        # plus clair que la photo, d'ou un bronze qui passait pour du platre.
        ao_img = getattr(mat, "occlusionTexture", None)
        aos.append(np.asarray(ao_img.convert("L"), dtype=np.float32) / 255.0
                   if ao_img is not None else None)

        vc = getattr(getattr(g, "visual", None), "vertex_colors", None)
        colors.append(np.asarray(vc, dtype=np.float32)[:, :3] / 255.0
                      if vc is not None else None)
        offset += len(g.vertices)

    if not verts:
        raise ValueError("no renderable geometry in the scene")

    tex = next((t for t in textures if t is not None), None)
    mr = next((t for t in mrs if t is not None), None)
    ao = next((t for t in aos if t is not None), None)
    vcol = next((c for c in colors if c is not None), None)
    return (np.concatenate(verts), np.concatenate(faces),
            np.concatenate(uvs), tex, mr, ao, vcol)


def render_turntable(
    glb_path: Path,
    out_path: Path,
    views: Iterable[float] = (30.0, 120.0, 210.0, 300.0),
    resolution: int = 512,
    background: float = 0.09,
) -> Path:
    """Render `views` azimuths side by side into one PNG contact sheet."""
    import trimesh
    from PIL import Image

    from . import uvraster

    scene = trimesh.load(str(glb_path), process=False)
    v, f, uv, tex, mr, ao, vcol = _collect(scene)

    centre = (v.min(0) + v.max(0)) / 2
    radius = float(np.linalg.norm(v.max(0) - v.min(0))) / 2
    v = v - centre

    dev = "cuda"
    # nvdiffrast antialiased the outline analytically, from the geometry. There
    # is no cheap equivalent in plain PyTorch, so the frame is rendered larger
    # and averaged down instead - a few tenths of a second on a thumbnail, and
    # it smooths the shading as well as the silhouette.
    big = resolution * _SUPERSAMPLE
    v_t = torch.tensor(v, device=dev)
    f_t = torch.tensor(f, device=dev, dtype=torch.int64)
    uv_t = torch.tensor(uv, device=dev)
    tex_t = torch.tensor(tex, device=dev) if tex is not None else None
    mr_t = torch.tensor(mr, device=dev) if mr is not None else None
    ao_t = torch.tensor(ao, device=dev)[..., None] if ao is not None else None
    vcol_t = torch.tensor(vcol, device=dev) if vcol is not None else None

    # Face normals smoothed onto vertices, for the lambert term.
    tri = v_t[f_t]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    vn = torch.zeros_like(v_t)
    vn.index_add_(0, f_t.reshape(-1), fn.repeat_interleave(3, dim=0))
    vn = torch.nn.functional.normalize(vn, dim=-1)

    proj = torch.tensor(_perspective(35.0, 1.0), device=dev)
    tiles = []

    for az in views:
        a = math.radians(az)
        dist = radius * 3.1
        eye = (math.sin(a) * dist, radius * 0.75, math.cos(a) * dist)
        view = torch.tensor(_look_at(eye, (0, 0, 0)), device=dev)
        mvp = proj @ view

        homo = torch.cat([v_t, torch.ones_like(v_t[:, :1])], dim=-1)
        clip = homo @ mvp.T

        # Everything the shading needs, interpolated in one pass: there is no
        # rasteriser state to hand back to, so the attributes travel with the
        # triangles rather than being fetched afterwards.
        carried = [vn]
        if tex_t is not None or mr_t is not None:
            carried.append(uv_t)
        if vcol_t is not None:
            carried.append(vcol_t)
        values, mask = uvraster.camera(clip, f_t, big, carried)

        normal = torch.nn.functional.normalize(values[0], dim=-1)
        st = None
        if tex_t is not None or mr_t is not None:
            st = values[1] % 1.0
            st = torch.stack([st[..., 0], 1.0 - st[..., 1]], dim=-1)

        if tex_t is not None:
            albedo = uvraster.sample(tex_t, st)
        elif vcol_t is not None:
            albedo = values[-1]
        else:
            albedo = torch.full((big, big, 3), 0.75, device=dev)

        # Metal-aware shading. Without it, a metallic asset renders as chalky
        # diffuse and reads as washed out even when its albedo is perfectly
        # saturated - which is exactly how a good crown got misjudged once.
        if mr_t is not None:
            mr = uvraster.sample(mr_t, st)
            rough = mr[..., 1:2].clamp(0.04, 1.0)   # glTF packs G=roughness
            metal = mr[..., 2:3].clamp(0.0, 1.0)    #                 B=metallic
        else:
            rough = torch.full_like(albedo[..., :1], 0.5)
            metal = torch.zeros_like(rough)

        # Convention glTF : l'occlusion attenue l'AMBIANT, jamais la lumiere
        # directe — une crevasse reste eclairee par le soleil qui la vise.
        occ = uvraster.sample(ao_t.expand(-1, -1, 3), st)[..., :1]             if ao_t is not None else torch.ones_like(albedo[..., :1])

        light = torch.nn.functional.normalize(
            torch.tensor([0.4, 0.7, 0.6], device=dev), dim=0)
        n_dot_l = (normal * light).sum(-1, keepdim=True).clamp(0, 1)
        hemi = 0.5 + 0.5 * normal[..., 1:2]

        # A sky/ground gradient standing in for an environment map: enough for
        # metal to pick up a bright top and a darker bottom instead of flat grey.
        sky = torch.tensor([0.62, 0.66, 0.74], device=dev)
        ground = torch.tensor([0.22, 0.20, 0.18], device=dev)
        env = ground + (sky - ground) * hemi

        view_dir = torch.tensor([0.0, 0.0, 1.0], device=dev)
        half = torch.nn.functional.normalize(light + view_dir, dim=0)
        n_dot_h = (normal * half).sum(-1, keepdim=True).clamp(0, 1)
        gloss = (2.0 / rough.pow(2) - 2.0).clamp(1.0, 2048.0)
        spec = n_dot_h.pow(gloss) * (1.0 - rough) * 1.6

        # Metals tint their reflection with the albedo and have no diffuse.
        f0 = 0.04 * (1 - metal) + albedo * metal
        diffuse = albedo * (1 - metal) * (
            (0.18 + 0.35 * env) * occ + 0.9 * n_dot_l)
        reflection = f0 * (env * (0.55 * occ + 0.45 * n_dot_l) + spec)

        shaded = (diffuse + reflection).clamp(0, 1) ** (1 / 2.2)

        hit = mask[..., None].float()
        img = shaded * hit + background * (1 - hit)
        # Average the supersampled frame down to size. Done on the composited
        # image rather than on coverage, so an edge pixel blends the object
        # into the background exactly as far as it is covered.
        img = torch.nn.functional.avg_pool2d(
            img.permute(2, 0, 1).unsqueeze(0), _SUPERSAMPLE)[0].permute(1, 2, 0)
        tiles.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    sheet = np.concatenate(tiles, axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("-r", "--resolution", type=int, default=512)
    args = ap.parse_args()
    out = args.out or args.glb.with_name(args.glb.stem + "_preview.png")
    print(render_turntable(args.glb, out, resolution=args.resolution))
