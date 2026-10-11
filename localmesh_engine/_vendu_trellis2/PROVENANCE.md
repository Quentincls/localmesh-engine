# Provenance

- **Œuvre** : TRELLIS.2 (code d'inférence), Microsoft Research.
- **Dépôt d'origine** : https://github.com/microsoft/TRELLIS.2
- **Licence** : MIT, Copyright (c) Microsoft Corporation (voir LICENSE à côté).
- **Chemin d'arrivée** : le sous-paquet `trellis2` a été copié depuis le nœud
  ComfyUI `visualbruno/ComfyUI-Trellis2` (https://github.com/visualbruno/ComfyUI-Trellis2,
  fichier LICENSE = MIT) qui l'embarquait (essai du 9 août 2026, voir DOCUMENTATION.md,
  « L'essai TRELLIS.2 »), puis adapté pour tourner hors de ComfyUI
  (`comfy_shim.py`, `patches.py`, `leurre_nvdiffrast.py`).
- **Modifications LocalMesh** : rasterisation par `localmesh_engine.uvraster` au lieu
  de nvdiffrast ; chargement fp8 ; plafond de tokens par carte ; pas de
  dépendance à ComfyUI. Les fichiers modifiés portent leurs commentaires.
- **Poids** : aucun poids n'est distribué avec ce dépôt. `TRELLIS.2-4B`
  est sous MIT (Microsoft) et la variante fp8 `visualbruno/TRELLIS.2-4B-FP8`
  déclare également MIT sur Hugging Face (relevé le 7 septembre 2026).
  Les adresses et les licences de chaque jeu sont dans le fichier NOTICE.
- **Voie multi-vue** (vérifié le 4 septembre 2026) : TRELLIS.2 de Microsoft
  n'a aucune voie multi-vue (`run()` prend une image). `run_multiview`,
  `sample_*_multiview`, `FlowEulerMultiViewSampler` et ses variantes RK4/RK5/
  Heun, le verrou DINO et le bouchage des trous sont des ajouts du nœud
  ComfyUI, distribués sous son fichier LICENSE (MIT). LocalMesh les a
  modifiés (axes du mélange posés sur le repère réel de TRELLIS.2, passage
  négatif partagé, prédictions au fil de l'eau) ; les fichiers portent leurs
  commentaires. Le moteur tourne hors de ComfyUI, appelé directement en Python.
- **Modifications de la version 1.2.0** (22 et 23 septembre 2026), deux
  fichiers, commentaires sur place :
  - `trellis2/pipelines/trellis2_image_to_3d.py` : `texture_mesh` et
    `texture_mesh_multiview` prennent `cond_resolution`, la finesse du
    conditionnement d'image détachée de la grille de forme. None rend le
    comportement d'avant. `texture_mesh` prend aussi `max_tex_tokens`, et
    `encode_shape_slat` prend `max_tokens` : les jetons de texture sont
    comptés sur le processeur avant l'encodeur, et au-delà du plafond la
    texture passe au DiT 512 au lieu de déborder de la carte. None rend le
    comportement d'avant.
  - `trellis2/pipelines/samplers/flow_euler.py` : le vote des vues de la
    texture multi-vue peut se normaliser sur l'étendue du sujet plutôt que
    sur la grille, sous `LUMENGEN_MV_TEX_ISO=1`. Inerte par défaut.
- **Un fichier retire** (24 septembre 2026) : `trellis2/utils/mesh_utils.py`.
  Il importait `plyfile` (GPL-3.0) au niveau module, et rien ne l'importait.
  Le tableau ci-dessous compte les fichiers d'avant ce retrait.

## Qui a ecrit quoi, fichier par fichier

Compare le 10 septembre 2026 aux arborescences des trois depots amont
(API GitHub, arbres recursifs). Les 86 fichiers Python de ce dossier se
repartissent ainsi :

| origine | fichiers | licence |
|---|---|---|
| `microsoft/TRELLIS.2` | 81 | MIT, Copyright (c) Microsoft Corporation |
| `TencentARC/Pixal3D` | 3 | MIT, Copyright (c) 2026 Tencent |
| `visualbruno/ComfyUI-Trellis2` | 1 | MIT (fichier LICENSE au nom de Microsoft) |
| LocalMesh | 1 | Apache-2.0 |

**Les trois fichiers de Tencent.** Ils n'existent pas chez Microsoft (verifie :
404 sur `raw.githubusercontent.com/microsoft/TRELLIS.2/main/...`) et existent
chez `TencentARC/Pixal3D` sous le prefixe `pixal3d/`. Ils sont vivants, tous
les trois importes :

- `trellis2/modules/attention/proj_attention.py`
- `trellis2/modules/sparse/attention/proj_attention.py`
- `trellis2/trainers/flow_matching/mixins/image_conditioned_proj.py`

Le MIT exige que l'avis de copyright voyage avec le code. Le texte de Tencent
est donc pose a cote, dans `LICENSE-Pixal3D`, et leur `NOTICE` complet dans
`NOTICE-Pixal3D`.

**Le fichier de visualbruno** : `trellis2/utils/camera.py`, present dans son
depot et absent de celui de Microsoft. Son fichier LICENSE porte le MIT au nom
de Microsoft Corporation, celui qui est deja a cote sous `LICENSE`.

**Le notre** : `trellis2/contexte.py`, qui remplace ce que le paquet ComfyUI
fournissait (trois chemins de modeles et une barre de progression), pour que
la bibliotheque tourne hors de ComfyUI.
