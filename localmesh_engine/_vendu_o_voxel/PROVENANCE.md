# Provenance

- **Œuvre** : o_voxel (O-Voxel), le post-traitement de TRELLIS.2 (cuisson du
  volume PBR dans des cartes UV), auteur Jianfeng Xiang, Microsoft Research.
- **Dépôt d'origine** : https://github.com/microsoft/TRELLIS.2 (sous-paquet
  `o-voxel`), roue `o_voxel-0.0.1` construite localement pour Windows /
  torch 2.8 (`engine/_wheels_src`).
- **Licence** : MIT, Copyright (c) Microsoft Corporation (voir LICENSE à côté).
- **Modifications LocalMesh** : `postprocess.py` réécrit sans nvdiffrast (la
  rasterisation passe par `localmesh_engine.uvraster`) ; le reste de la roue n'est
  pas copié ici.
