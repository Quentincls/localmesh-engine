"""Recaler finement chaque photo sur le modèle en comparant RELIEF contre RELIEF.

La projection place une caméra rigide par photo. Là où la forme générée
diffère un peu de la photo, par exemple une tête tournée de quelques degrés de
plus, le nez de la photo tombe à côté du nez du modèle, et le visage sort en
double. Ce module rend, pour chaque vue, un CHAMP DE DÉPLACEMENTS qui dit où
lire la photo pour que son relief tombe sur celui du modèle.

POURQUOI LE RELIEF, ET PAS LES COULEURS. Mesuré le 23 septembre 2026 :
- un flot optique entre la photo et la texture DESSINÉE échoue, car le dessin
  invente son contenu ; il n'était fiable que sur 9 à 17 % des pixels ;
- les traits DINOv3 appariés entre ces deux images sont justes à un jeton
  près, mais bruités de 5 à 12 px : un champ tiré d'eux ondulait le texte de la
  cassette, pourtant déjà exacte ;
- Depth Anything 3 (base) voit un objet détouré presque plat.
MoGe-2 (Microsoft, MIT) rend l'ORIENTATION de la surface en chaque pixel de la
photo, avec le détail d'un nez ou d'une vis. Comparée à l'orientation du
modèle rendu de la même caméra, c'est la même sorte d'image des deux côtés :
un flot optique entre elles devient fiable.

LES GARDE-FOUS, SANS LESQUELS LA PHOTO SE TORD. Un déplacement n'est gardé que
s'il est confirmé dans les deux sens, s'il rend les deux reliefs vraiment
semblables, et s'il améliore NETTEMENT leur accord sur une zone large. Sans ce
dernier critère, les fines rayures du modèle attiraient le flot et tordaient
le globe imprimé de la cassette. Avec lui, la cassette ne bouge plus d'un
pixel, et le second profil de la statue disparaît.

Les poids de MoGe-2 ne sont JAMAIS téléchargés d'ici : sans eux, le module se
dit indisponible et la projection garde sa caméra rigide.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Le dépôt des poids, celui que l'application utilise déjà pour ses normales.
DEPOT = "Ruicheng/moge-2-vitl-normal"


@dataclass
class ReglagesRelief:
    """Chaque valeur a été mesurée sur douze sujets, cassette comprise."""
    #: La grille de travail, en pixels.
    cote: int = 512
    #: Le flot le plus long qu'on accepte, en pixels de la grille.
    max_px: float = 24.0
    #: L'écart maximal entre l'aller et le retour du flot.
    coherence: float = 2.0
    #: Après déplacement, les deux reliefs doivent coïncider à tant de degrés.
    angle_ok: float = 22.0
    #: Le gain d'accord exigé, en degrés, moyenné sur une zone de ce rayon.
    gain_min: float = 8.0
    sigma_gain: float = 8.0
    #: Le lissage du champ gardé, et l'appui en dessous duquel il s'éteint.
    sigma: float = 4.0
    appui: float = 0.35


# ------------------------------------------------------------------ MoGe ----
def _poids():
    """Le fichier de poids de MoGe-2, s'il est déjà sur le disque."""
    from . import config
    local = Path(config.MODELS_ROOT) / "Ruicheng" / "moge-2-vitl-normal" / "model.pt"
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(DEPOT, "model.pt", local_files_only=True))
    except Exception:                                         # noqa: BLE001
        return None


def disponible() -> bool:
    return _poids() is not None


def _shim_utils3d() -> None:
    """Le runtime porte l'utils3d de l'ère TRELLIS, sans le sous-module `pt`
    que MoGe attend. Les deux fonctions qu'il utilise tiennent en vingt lignes
    (même contournement que l'application, `lumengen/mogenormals.py`)."""
    import torch
    import utils3d
    if sys.modules.get("utils3d.pt") is not None:
        return
    pt = types.ModuleType("utils3d.pt")

    def intrinsics_from_focal_center(fx, fy, cx, cy):
        fx, fy, cx, cy = torch.broadcast_tensors(
            *(torch.as_tensor(v, dtype=torch.float32) for v in (fx, fy, cx, cy)))
        z, o = torch.zeros_like(fx), torch.ones_like(fx)
        return torch.stack([torch.stack([fx, z, cx], -1), torch.stack([z, fy, cy], -1),
                            torch.stack([z, z, o], -1)], -2)

    def depth_map_to_point_map(depth, intrinsics=None):
        h, w = depth.shape[-2:]
        j = (torch.arange(w, device=depth.device, dtype=depth.dtype) + 0.5) / w
        i = (torch.arange(h, device=depth.device, dtype=depth.dtype) + 0.5) / h
        v, u = torch.meshgrid(i, j, indexing="ij")
        x = (u - intrinsics[..., 0, 2]) / intrinsics[..., 0, 0] * depth
        y = (v - intrinsics[..., 1, 2]) / intrinsics[..., 1, 1] * depth
        return torch.stack([x, y, depth], dim=-1)

    pt.intrinsics_from_focal_center = intrinsics_from_focal_center
    pt.depth_map_to_point_map = depth_map_to_point_map
    utils3d.pt = pt
    sys.modules["utils3d.pt"] = pt


def _charger():
    # xformers du runtime : compilé pour une autre génération de carte.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    _shim_utils3d()
    parent = str(Path(__file__).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import importlib
    v2 = importlib.import_module("localmesh_engine._vendu_moge.model.v2")
    poids = _poids()
    if poids is None:
        # Jamais `from_pretrained("None")` : il partirait chercher sur le réseau.
        raise RuntimeError("MoGe-2 weights are missing. Skipping depth alignment.")
    return v2.MoGeModel.from_pretrained(str(poids)).cuda().eval()


def normales_des_photos(photos):
    """{rôle: image H x W x 3 dans 0..1} -> {rôle: normales H x W x 3}, dans le
    repère de l'appareil de MoGe (x à droite, y en bas, z vers la scène)."""
    import torch
    modele = _charger()
    try:
        sortie = {}
        for r, im in photos.items():
            t = torch.as_tensor(im, dtype=torch.float32).permute(2, 0, 1).cuda()
            with torch.no_grad():
                sortie[r] = modele.infer(t, apply_mask=False, use_fp16=True)["normal"].float().cpu().numpy()
        return sortie
    finally:
        del modele
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- le flot ----
_LUMIERES = [np.array(v, np.float32) / np.linalg.norm(v)
             for v in ((-0.5, -0.5, -1.0), (0.6, -0.2, -1.0), (0.0, 0.7, -1.0))]


def _eclairer(n, m, l):
    s = np.clip((n * l).sum(-1), 0, 1)
    s[~m] = 0
    return (s * 255).astype(np.uint8)


def _lire(img, f):
    """img(x + f(x)), bilinéaire."""
    import cv2
    H, W = f.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def _flou(v, w, sigma):
    """Moyenne gaussienne de `v` pondérée par `w`."""
    import cv2
    k = int(3 * sigma) * 2 + 1
    num = cv2.GaussianBlur((v * w).astype(np.float32), (k, k), sigma)
    den = cv2.GaussianBlur(w.astype(np.float32), (k, k), sigma)
    return num, den


def champ(normales_photo, normales_modele, masque_photo, masque_modele,
          rg: ReglagesRelief | None = None):
    """Le champ de déplacements (H x W x 2, en pixels de la grille) qui dit où
    lire la photo pour que son relief tombe sur celui du modèle, et la part de
    la silhouette qu'il déplace. Nul là où rien n'est sûr."""
    import cv2
    rg = rg or ReglagesRelief()
    npx, nm = normales_photo.astype(np.float32), normales_modele.astype(np.float32)
    mp, mr = masque_photo.astype(bool), masque_modele.astype(bool)
    # MoGe et le moteur n'ont pas la même convention d'axes : on retourne
    # l'axe qui s'oppose en moyenne là où les deux voient l'objet.
    m = mp & mr
    if m.sum() < 500:
        return np.zeros(npx.shape[:2] + (2,), np.float32), 0.0
    for k in range(3):
        if np.median(npx[m][:, k] * nm[m][:, k]) < 0 and abs(np.median(nm[m][:, k])) > 0.1:
            npx[..., k] *= -1
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    aller, retour = [], []
    for l in _LUMIERES:
        a, b = _eclairer(nm, mr, l), _eclairer(npx, mp, l)
        aller.append(dis.calc(a, b, None))
        retour.append(dis.calc(b, a, None))
    f = np.median(np.stack(aller), 0).astype(np.float32)
    r = np.median(np.stack(retour), 0).astype(np.float32)
    ecart = np.hypot(*(f + _lire(r, f)).transpose(2, 0, 1))
    npx_d = _lire(npx, f)
    mp_d = _lire(mp.astype(np.float32), f) > 0.5
    a0, a1 = _angle(npx, nm), _angle(npx_d, nm)
    n0, d0 = _flou(a0, m, rg.sigma_gain)
    n1, d1 = _flou(a1, mr & mp_d, rg.sigma_gain)
    gain = n0 / np.maximum(d0, 1e-6) - n1 / np.maximum(d1, 1e-6)
    sur = (mr & mp_d & (ecart < rg.coherence) & (a1 < rg.angle_ok)
           & (np.hypot(f[..., 0], f[..., 1]) < rg.max_px) & (gain > rg.gain_min))
    w = sur.astype(np.float32)
    nx, den = _flou(f[..., 0], w, rg.sigma)
    ny, _ = _flou(f[..., 1], w, rg.sigma)
    garde = den / (den + rg.appui)
    out = np.stack([nx, ny], -1) / np.maximum(den, 1e-6)[..., None] * garde[..., None]
    out[~mr] = 0
    return out.astype(np.float32), float(sur[mr].mean())


def normales_rendues(xyz, couvert, yaw, elev, roul):
    """Les normales du maillage vu de la caméra, depuis le tampon des points 3D
    rendus, dans le repère de l'appareil (x à droite, y en bas, z vers la scène,
    tournées vers l'appareil)."""
    X = xyz.astype(np.float32)
    du = np.zeros_like(X); dv = np.zeros_like(X)
    du[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2
    dv[1:-1] = (X[2:] - X[:-2]) / 2
    n = np.cross(du, dv)
    cy, sy, ce, se = math.cos(yaw), math.sin(yaw), math.cos(elev), math.sin(elev)
    x1 = cy * n[..., 0] - sy * n[..., 2]
    z1 = sy * n[..., 0] + cy * n[..., 2]
    y2 = ce * n[..., 1] - se * z1
    z2 = se * n[..., 1] + ce * z1
    cr, sr = math.cos(roul), math.sin(roul)
    nc = np.stack([cr * x1 + sr * y2, sr * x1 - cr * y2, -z2], -1)
    nc /= np.maximum(np.linalg.norm(nc, axis=-1, keepdims=True), 1e-9)
    nc = np.where(nc[..., 2:] > 0, -nc, nc)
    nc[~couvert] = 0
    return nc
