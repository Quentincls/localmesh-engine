# -*- coding: utf-8 -*-
"""Position tournante axiale, pour un plan d'image.

Écrit pour LocalMesh Engine, sous Apache-2.0, à partir des mathématiques
publiées de RoFormer (Su et al., 2021, arXiv:2104.09864) étendues aux deux
axes d'une image. Il remplace un fichier repris de Meta qui portait l'accord
de licence DINOv3 : cet accord n'est pas une licence libre, il pose des
restrictions d'usage, et il aurait obligé tout le dépôt à voyager avec lui.
Ces cent soixante lignes de trigonométrie ne valent pas cette dette.

Le principe. Chaque cellule (h, w) reçoit deux coordonnées ramenées dans
[-1, +1]. Pour un jeu de périodes en progression géométrique, on en tire un
angle par période et par axe. Le vecteur de traits est alors coupé en deux
moitiés, et chaque paire (première moitié, seconde moitié) tourne de cet
angle. Deux cellules voisines subissent des rotations proches : l'attention
qui suit lit la POSITION dans le produit scalaire, sans qu'aucun poids ne
l'ait apprise.

La disposition des angles n'est pas libre : le tenseur `periods` est chargé
depuis le point de contrôle, et la moitié haute du vecteur doit tourner
contre la moitié basse dans le même ordre qu'à l'entraînement. C'est
pourquoi les angles sont rangés [u0..u_{D/4-1}, v0..v_{D/4-1}] puis répétés
deux fois : cet ordre-là fait partie du format des poids.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn

DEUX_PI = 2.0 * math.pi


def tourner_moities(x: Tensor) -> Tensor:
    """Le quart de tour qui accompagne le cosinus.

    [a, b] devient [-b, a], les deux moitiés étant prises sur la dernière
    dimension. Combiné à `x * cos`, cela donne la rotation plane de chaque
    paire (a_i, b_i) sans jamais construire de matrice.
    """
    haut, bas = x.chunk(2, dim=-1)
    return torch.cat((-bas, haut), dim=-1)


class RoPE(nn.Module):
    """Fait tourner un plan de traits selon la position de chaque cellule.

    L'entrée est une carte `[B, num_heads * D_tete, H, W]` ; la sortie a la
    même forme (`layout="spatial"`) ou la forme séparée par tête
    `[B, H, W, num_heads, D_tete]` (`layout="flatten"`), qui est celle que
    l'attention de voisinage attend.

    `base` et le couple `min_period` / `max_period` sont exclusifs : le
    premier engendre des périodes en puissances, le second les répartit
    régulièrement sur une échelle logarithmique entre deux bornes.

    Les trois perturbations de coordonnées (`shift_coords`, `jitter_coords`,
    `rescale_coords`) n'agissent qu'en apprentissage. En inférence elles ne
    font rien, et le moteur n'appelle jamais ce module autrement.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError(
                "embed_dim doit etre un multiple de 4 * num_heads : "
                "chaque tete range D/4 periodes sur deux axes, puis double."
            )
        bornes = min_period is not None and max_period is not None
        if (base is None) == (not bornes):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.num_heads = num_heads
        self.D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        # Les coordonnees ne dependent que de la taille de la carte : on les
        # garde d'un appel a l'autre, et on ne les recalcule qu'au changement.
        self._cached_coords: Optional[Tensor] = None
        self._cached_hw: Optional[tuple] = None

        # `persistent=True` : ce tenseur EST dans le point de controle. Le
        # nom `periods` fait donc partie du format des poids et ne peut pas
        # changer.
        self.register_buffer(
            "periods",
            torch.empty(self.D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    # -- les periodes ------------------------------------------------------
    def _init_weights(self) -> None:
        n = self.D_head // 4
        ou = {"device": self.periods.device, "dtype": self.dtype}
        if self.base is not None:
            # base^(0), base^(2/ (D/2)), base^(4/(D/2)) ... : une progression
            # geometrique, la meme construction que le RoPE d'origine.
            exposants = 2.0 * torch.arange(n, **ou) / (self.D_head // 2)
            periodes = self.base ** exposants
        else:
            periodes = torch.logspace(
                math.log10(self.min_period), math.log10(self.max_period), steps=n)
        self.periods.data = periodes

    # -- les coordonnees ---------------------------------------------------
    def create_coordinate(self, *, H: int, W: int) -> Tensor:
        """Le centre de chaque cellule, ramene dans [-1, +1], en [HW, 2]."""
        ou = {"device": self.periods.device, "dtype": self.dtype}
        centres_h = torch.arange(0.5, H, **ou)
        centres_w = torch.arange(0.5, W, **ou)

        if self.normalize_coords == "separate":
            # chaque axe est ramene par SA propre taille : un rectangle est
            # traite comme un carre etire
            centres_h = centres_h / H
            centres_w = centres_w / W
        elif self.normalize_coords == "max":
            grand = float(max(H, W))
            centres_h = centres_h / grand
            centres_w = centres_w / grand
        elif self.normalize_coords == "min":
            petit = float(min(H, W))
            centres_h = centres_h / petit
            centres_w = centres_w / petit
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        grille_h, grille_w = torch.meshgrid(centres_h, centres_w, indexing="ij")
        coords = torch.stack((grille_h, grille_w), dim=-1).flatten(0, 1)  # [HW, 2]
        coords = coords.mul(2.0).sub(1.0)

        # Les trois perturbations d'apprentissage. En inference, aucune.
        if self.training and self.shift_coords is not None:
            d = self.shift_coords
            coords = coords + torch.empty(2, **ou).uniform_(-d, d)[None, :]
        if self.training and self.jitter_coords is not None:
            m = math.log(self.jitter_coords)
            coords = coords * torch.empty(2, **ou).uniform_(-m, m).exp()[None, :]
        if self.training and self.rescale_coords is not None:
            m = math.log(self.rescale_coords)
            coords = coords * torch.empty(1, **ou).uniform_(-m, m).exp()
        return coords

    # -- la rotation -------------------------------------------------------
    def rotate(self, x: Tensor, coords: Tensor) -> Tensor:
        # [HW, 2, D/4] : un angle par axe et par periode
        angles = DEUX_PI * coords[:, :, None] / self.periods[None, None, :]
        # a plat : [u0..u_{D/4-1}, v0..v_{D/4-1}], soit D/2 angles
        angles = angles.flatten(1, 2)
        # repete pour couvrir la dimension entiere : la seconde moitie du
        # vecteur tourne avec les MEMES angles que la premiere
        angles = angles.tile(2)
        return x * angles.cos() + tourner_moities(x) * angles.sin()

    # -- l'appel -----------------------------------------------------------
    def forward(self, x: Tensor, layout: str = "spatial") -> Tensor:
        b, _, h, w = x.shape
        # [B, n*d, H, W] -> [B, n, HW, d] : les tetes tournent chacune de leur cote
        x = x.reshape(b, self.num_heads, self.D_head, h * w).transpose(-1, -2)

        if (h, w) != self._cached_hw:
            self._cached_coords = self.create_coordinate(H=h, W=w)
            self._cached_hw = (h, w)

        x = self.rotate(x, self._cached_coords)

        if layout == "spatial":
            return x.transpose(-1, -2).reshape(b, self.num_heads * self.D_head, h, w)
        if layout == "flatten":
            return x.reshape(b, self.num_heads, h, w, self.D_head).permute(0, 2, 3, 1, 4)
        raise ValueError(f"Unknown layout: {layout}")
