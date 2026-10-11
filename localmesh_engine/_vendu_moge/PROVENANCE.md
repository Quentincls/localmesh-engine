# Provenance

- **Œuvre** : MoGe (Monocular Geometry Estimation), Microsoft Research.
- **Dépôt d'origine** : https://github.com/microsoft/MoGe
- **Licence** : MIT, Copyright (c) Microsoft Corporation (voir LICENSE à côté).
- **Sous-arbre `model/dinov2/`** : copie du DINOv2 de Meta embarquée par MoGe,
  Apache License 2.0, Copyright (c) Meta Platforms, Inc. and affiliates
  (voir `model/dinov2/LICENSE`). Ses fichiers citent eux-mêmes leurs sources
  (DINO, timm).
- **Copie** : prise telle quelle dans l'application LocalMesh
  (`lumengen/_vendu_moge`), elle-même copie du dépôt d'origine.
- **Usage dans le moteur** : `relief.py`, les normales lues dans chaque
  photographie pour recaler finement la projection des photos sur le maillage.
- **Poids** : `Ruicheng/moge-2-vitl-normal` (MIT). Le moteur ne les télécharge
  JAMAIS : il les lit dans `models/Ruicheng/moge-2-vitl-normal/model.pt` ou
  dans le cache Hugging Face, et sans eux le recalage fin est sauté.
- **Environnement** : le sous-module `utils3d.pt` que ce code attend est fourni
  par `relief._shim_utils3d` (deux fonctions), et xformers est désactivé.
