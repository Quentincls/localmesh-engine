# Weights

Nothing in this repository is a model file. The engine reads its weights from
one folder on disk and, with one exception documented below, never fetches them
for you. This page says where each set comes from, where it goes, what it weighs
and under which licence. Source files are named as it goes; they all live under
`localmesh_engine/`, the package folder of this repository.

The root of that folder is the environment variable `LOCALMESH_ROOT`;
`LUMENGEN_ROOT` is read after it, for installations that predate the name
(`config.py`). Plan for about 14.3 GB, plus the Hugging Face cache: 9.5 GB for
the single photo path, 4.8 GB more for the four view path.

## The layout

Exact names; the engine does not search for alternatives.

```
<LOCALMESH_ROOT>/models/
  TRELLIS.2-4B/                                         8.1 GB
    pipeline.json            or  pipeline_fp8.json      both names accepted
    pipeline.lumengen.json                              written by the engine
    ckpts_fp8/                                          18 files, .json + .safetensors
  microsoft/TRELLIS-image-large/ckpts/                  148 MB
    ss_dec_conv3d_16l8_fp16.safetensors  + .json
  facebook/dinov3-vitl16-pretrain-lvd1689m/             1.2 GB, manual access
    model.safetensors, config.json, preprocessor_config.json
  multivue/                                             4.8 GB, four view path
    structure_mv_fp8.safetensors   + .json              1.39 GB
    forme_512_mv_fp8.safetensors   + .json              1.44 GB
    forme_1024_mv_fp8.safetensors  + .json              1.44 GB
    champ.safetensors                                   2.7 MB
    cameras/                                            544 MB
      model.safetensors, config.json
      source/     depth-anything-3 checkout
      deps/       omegaconf, addict, antlr4
  hf/                                                   Hugging Face cache
```

Each `*_mv_fp8.safetensors` needs its `.json`: it describes the architecture, and
without it `Poids.manquants` reports the file as missing. `pipeline.lumengen.json`
is not yours to provide, the engine writes it on first load: a side-car copy of
the descriptor with the background remover swapped for BiRefNet_HR.

## The sets

| Set | Size | Source | Licence | Required |
|---|---|---|---|---|
| TRELLIS.2-4B, FP8 | 8.1 GB | [visualbruno/TRELLIS.2-4B-FP8](https://huggingface.co/visualbruno/TRELLIS.2-4B-FP8) | MIT | yes |
| Sparse structure decoder | 148 MB | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | MIT | yes |
| DINOv3 ViT-L/16 | 1.2 GB | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License, Meta | yes, manual access |
| BiRefNet_HR | 444 MB | [ZhengPeng7/BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR) | MIT | yes, fetched automatically |
| Multi-view structure and shape, FP8 | 4.27 GB | [Qtn-Cls/LocalMeshEngine](https://huggingface.co/Qtn-Cls/LocalMeshEngine) | MIT, Copyright (c) 2026 Tencent | four view path |
| Field network (NAF) | 2.7 MB | [Qtn-Cls/LocalMeshEngine](https://huggingface.co/Qtn-Cls/LocalMeshEngine) | Apache-2.0, valeo.ai | four view path |
| Side measurement (DA3-BASE) | 544 MB | [depth-anything/DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | Apache-2.0 | four view path, no hard stop without it |

Commands below use `hf`, the command shipped with `huggingface_hub`; the package
pins `huggingface_hub>=1.0`, where `hf` is the only CLI. They are written for a
POSIX shell. In Windows PowerShell, write `$env:LOCALMESH_ROOT` in place of
`$LOCALMESH_ROOT`, and put each command on a single line.

### TRELLIS.2-4B, FP8

The generator itself: sparse structure flow, both shape cascades, both texture
cascades, their encoders and decoders. `config.py` reads the folder name
literally, so keep it.

```
hf download visualbruno/TRELLIS.2-4B-FP8 \
  --local-dir "$LOCALMESH_ROOT/models/TRELLIS.2-4B"
```

### Sparse structure decoder

The FP8 set above does not carry this decoder; the engine reads it here.

```
hf download microsoft/TRELLIS-image-large \
  ckpts/ss_dec_conv3d_16l8_fp16.json \
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  --local-dir "$LOCALMESH_ROOT/models/microsoft/TRELLIS-image-large"
```

### DINOv3 ViT-L/16

The image encoder. Both paths need it, the single photo path and the four view
path (`multivue/traits.py`), and both read this local folder. Nothing fetches it
at run time, and without it the first generation stops.

**This repository is gated, and the gate is manual.** You need a Hugging Face
account, you request access on the model page, and a human at Meta approves it.
Start this before anything else: it is the one step that cannot be shortened.

```
hf auth login
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir "$LOCALMESH_ROOT/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
```

The DINOv3 License is not an open source licence. Two obligations follow, for
you as much as for us: display **"Built with DINOv3"** somewhere visible (a
website, a user interface, an about page, product documentation); and if you
redistribute the weights, ship a copy of the agreement and bind the recipient to
the same terms. We do not redistribute them; the notice is due anyway, and it
stands at the top of `NOTICE`. Full text:
<https://ai.meta.com/resources/models-and-libraries/dinov3-license/>. The copy
in the model repository, `LICENSE.md`, sits behind the same gate as the weights
and returns HTTP 401 until access is granted.

### Multi-view weights, hosted by us

Three FP8 files plus the field network, on [Qtn-Cls/LocalMeshEngine](https://huggingface.co/Qtn-Cls/LocalMeshEngine).
That repository is flat: its seven files sit at the root, and the command below
lands them exactly where the engine looks. There is nothing to move afterwards.

```
hf download Qtn-Cls/LocalMeshEngine \
  --local-dir "$LOCALMESH_ROOT/models/multivue"
```

`structure_mv_fp8`, `forme_512_mv_fp8` and `forme_1024_mv_fp8` are our own FP8
conversions of the official BF16 weights of
[TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) (MIT, Copyright
(c) 2026 Tencent), revision `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`, files
`ckpts/ss_flow_img_dit_1_3B_64_bf16_mv`,
`ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv` and
`ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv`. The alternative is the
originals: 16.5 GB against 4.27 GB, and the conversion left to you.

How they were converted. The transformer torso is cast to `float8_e4m3fn`, 480
tensors per file; the input and output layers, the normalisations and the
modulation stay in float32, 220 tensors; the complex RoPE buffer of the structure
model stays complex64. No trained value is otherwise touched. Each file carries
its source revision and the SHA-256 of its source file in its safetensors
metadata, under the keys `source_revision` and `source_sha256`. Compare
`source_sha256` with the LFS oid Hugging Face publishes for that file at that
revision: they match, so the chain is checkable without asking us. The
conversion is lossy and measured: weight RMSE 0.025 to 0.026, projection probe
0.027.

`champ.safetensors` is the official NAF checkpoint of
[valeoai/NAF](https://github.com/valeoai/NAF), from
`releases/download/model/naf_release.pth`, unmodified, stored as float32
safetensors so the engine can load it with `load_file`. Apache-2.0.

These four files serve the `draft`, `standard` and `high` tiers, see
[RECIPES.md](RECIPES.md). The four view path also needs `natten`, a compiled CUDA
extension imported by the field network
(`multivue/champ/layers/attentions.py`). It is code, not
weights: see step 5 of [INSTALL.md](INSTALL.md).

### Side measurement, DA3-BASE

This one reads the shooting angle of each photograph and decides which side each
profile actually shows, rather than trusting the slot it was dropped into.
Assembled by hand: weights, config, the upstream source tree, and three pure
Python packages placed beside it rather than in your environment.

```
hf download depth-anything/DA3-BASE model.safetensors config.json \
  --local-dir "$LOCALMESH_ROOT/models/multivue/cameras"

git clone https://github.com/ByteDance-Seed/depth-anything-3 \
  "$LOCALMESH_ROOT/models/multivue/cameras/source"

pip install --target "$LOCALMESH_ROOT/models/multivue/cameras/deps" \
  omegaconf==2.3.0 addict==2.4.0 antlr4-python3-runtime==4.9.3
```

`multivue/cotes.py` adds `cameras/source/src` and `cameras/deps` to `sys.path` at
call time, which is why nothing lands in your environment, and `manquants()`
names what is absent: the weights, the config, or the code. The code has two
accepted forms: this tree beside the weights, or an importable
`depth_anything_3` in your environment. The tree beside the weights wins when
both are there, because that is the copy the engine was measured against. Treat
this set as required: without it the engine falls back to the convention and
infers which way round the two profiles are, which mirrors the subject when the
inference is wrong. The result says which of the two happened.

## What the engine downloads on its own

One piece, and only one: **BiRefNet_HR**, the background remover. `matting.py`
names it as a Hugging Face repository id, not a local path, and loads it on the
first generation if the cache is empty. It is loaded with
`trust_remote_code=True`, which means the model's own Python is fetched and
executed in your process. Everything else must be in place before you start.

The vendored TRELLIS.2 tree can reach for two more, MoGe-2 and NAF, on the
Pixal3D recipe alone, which none of the four tiers selects and the command line
does not offer. `NOTICE` says where each comes from and under which licence.

`config.py` points `HF_HOME` at `<LOCALMESH_ROOT>/models/hf` unless you set it
yourself, so the download lands under the runtime root, and it turns off the
symlinks `huggingface_hub` normally creates: making one on Windows requires
developer mode or administrator rights.

## Preparing a machine with no network

Fetch on a connected machine, with `HF_HOME` already pointing at the runtime
root, then copy `<LOCALMESH_ROOT>/models/` across whole.

```
export HF_HOME="$LOCALMESH_ROOT/models/hf"
export HF_HUB_DISABLE_SYMLINKS=1
hf download ZhengPeng7/BiRefNet_HR
```

That pre-populates the one cache the engine reaches for. The second export is
the guard `config.py` sets for itself and `hf` does not: on Windows, without it,
the download can stop on `WinError 1314`. On the offline machine, set
`HF_HUB_OFFLINE=1`: any attempt to reach the Hub then fails immediately instead
of stalling. `trust_remote_code` still executes the Python that came with
BiRefNet_HR, now from your cache. Read it once if that matters to you.

## Two names for the model descriptor

`models/TRELLIS.2-4B/` must contain either `pipeline.json` or
`pipeline_fp8.json`. The engine accepts both (`pipeline.py`, `_NOMS_DESCRIPTEUR`).
Both exist because the original Microsoft repository writes the first name and
the FP8 conversion published on Hugging Face writes the second. Their contents
differ: `pipeline.json` on microsoft/TRELLIS.2-4B names `ckpts/..._bf16`,
`pipeline_fp8.json` on visualbruno/TRELLIS.2-4B-FP8 names `ckpts_fp8/..._fp8`.
Each descriptor belongs to the weights it shipped with. Accepting both names is
what lets you download a repository and use it as it came: rename nothing, and
do not mix the two. If both files are present, `pipeline.json` is the one read.
