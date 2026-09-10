# -*- coding: utf-8 -*-
"""Ce que les quatre photos donnent aux modèles.

Deux encodages par vue, et ils ne servent pas à la même chose.

Le premier est l'encodeur d'image habituel : il rend cinq jetons globaux, qui
disent ce qu'est l'objet, et une carte de jetons, qui dit où se trouve quoi.
C'est lui qui conditionne la structure et la forme.

Le second est le réseau de champ. Il rend une carte de requêtes à la
résolution de l'image, qui sert plus tard à rattraper le détail que la carte
de jetons a perdu : la carte de jetons est seize fois plus grossière que la
photo, et sans cette moitié haute fréquence les petites pièces disparaissent.

LES DEUX MODÈLES NE SONT JAMAIS RÉSIDENTS EN MÊME TEMPS. On encode les quatre
vues avec le premier, on le rend, on charge le second. C'est ce qui permet à
cette étape de tenir sur une carte de 8 Go.
"""
from __future__ import annotations

import gc
from pathlib import Path

#: Le côté de la carte de jetons : l'encodeur découpe l'image en tuiles de 16.
_TUILE = 16
#: La taille des requêtes du réseau de champ, fixée par son entraînement.
_REQUETES = (512, 512)


def encoder(config, images: dict, poids, taille: int = 512) -> dict:
    """Encoder les quatre vues préparées.

    `images` porte, par rôle, un tableau HxWx3 en flottants de 0 à 1, déjà
    détouré et mis au carré par la chaîne habituelle.

    Rend, par rôle, `{"jetons": ..., "requetes": ...}`. Les requêtes sortent
    toujours en 512 par 512, quelle que soit la taille d'entrée : c'est la
    résolution à laquelle le réseau a été entraîné.
    """
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file

    from .champ.model.naf import NAF

    roles = list(images)
    carre = {}
    for role, tableau in images.items():
        vue = Image.fromarray((np.clip(tableau, 0, 1) * 255).astype("uint8"))
        vue = vue.convert("RGB").resize((taille, taille), Image.Resampling.LANCZOS)
        carre[role] = np.asarray(vue, dtype=np.float32) / 255.0

    sortie = {role: {} for role in roles}

    # --- 1. l'encodeur d'image -------------------------------------------
    extracteur = config.trellis("modules.image_feature_extractor").DinoV3FeatureExtractor(
        str(config.MODELS_ROOT / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"),
        image_size=taille)
    extracteur.model = extracteur.model.to(dtype=torch.bfloat16, device="cuda").eval()
    attendu = (1, 5 + (taille // _TUILE) ** 2, 1024)
    for role in roles:
        tenseur = torch.from_numpy(carre[role].transpose(2, 0, 1).copy())[None].cuda()
        with torch.inference_mode():
            traits = extracteur(tenseur)
        tableau = traits.float().cpu().numpy()
        if tableau.shape != attendu or not np.isfinite(tableau).all():
            raise RuntimeError(
                "View %s: image encoder returned %s, expected %s"
                % (role, tableau.shape, attendu))
        sortie[role]["jetons"] = tableau
        del traits, tenseur, tableau
    del extracteur
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. le réseau de champ -------------------------------------------
    champ = NAF().eval()
    champ.load_state_dict(load_file(str(poids.champ)), strict=True)
    champ = champ.cuda()
    for role in roles:
        image = torch.from_numpy(carre[role].transpose(2, 0, 1).copy())[None].cuda()
        with torch.inference_mode():
            requetes = champ.image_encoder(image, output_size=_REQUETES)
        torch.cuda.synchronize()
        if tuple(requetes.shape) != (1, 256) + _REQUETES:
            raise RuntimeError(
                "View %s: field network returned %s" % (role, tuple(requetes.shape)))
        if not bool(torch.isfinite(requetes).all()):
            raise RuntimeError("View %s: field network returned non-finite values" % role)
        sortie[role]["requetes"] = requetes.cpu().numpy()
        del image, requetes
        gc.collect()
        torch.cuda.empty_cache()
    del champ
    gc.collect()
    torch.cuda.empty_cache()
    return sortie


def jetons_seuls(traits: dict) -> dict:
    """Les jetons d'image, sans les requêtes : ce que la structure attend."""
    return {role: v["jetons"] for role, v in traits.items()}
