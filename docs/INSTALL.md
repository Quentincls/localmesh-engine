# Installing LocalMesh Engine

Ten steps, numbered 0 to 9. At the end you run one command and get a textured `.glb`.

Reference machine: Windows, RTX 4060 Laptop with 8 GB, Python 3.12.13, torch
2.8.0+cu128, CUDA 12.8. The code itself is portable, and Linux should work, but
the three CUDA extensions of step 3 have to be built there and we have not
measured that path.

Read step 0 first. It starts a wait nobody can shorten.

## 0. Ask for DINOv3 access now

The image encoder is `facebook/dinov3-vitl16-pretrain-lvd1689m`. Meta gates that
repository as "manual": you need a Hugging Face account, you accept the licence
on the model page, you request access, and a human at Meta approves it. Nothing
generates without those files, on either path.

https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

Send the request, then install everything else while you wait.

Using DINOv3 comes with an obligation: display "Built with DINOv3" somewhere
visible, and ship a copy of the agreement with any redistribution of the
weights.

## 1. What the machine needs

| | |
|---|---|
| Python | 3.12 exactly. The package declares `>=3.12,<3.13`, and the CUDA extensions are cp312 builds |
| GPU | one NVIDIA card, 8 GB of video memory |
| Driver and toolkit | a driver for CUDA 12.8, plus the CUDA 12.8 toolkit with `nvcc`, which step 3 compiles against. `nvcc --version` must answer before you start step 3 |
| Compiler | MSVC Build Tools on Windows, gcc on Linux |
| git | the three extensions of step 3 are cloned with their submodules, and one dependency, `utils3d`, installs from a pinned git revision |
| Disk | about 9.5 GB of weights for the single photo path, 14.3 GB with the four views |

The four quality tiers are `draft`, `standard`, `high` and `max`; those four
identifiers are what `--tier` takes, and they are written Draft, Standard,
Detailed and Extreme where these pages spell them out. Three of the four
declare no memory floor, so an 8 GB card serves `draft`, `standard` and `high`.
`max` asks for 24 GB and refuses itself below that.

## 2. torch and torchvision first, pinned

Clone the repository first. Every command from here on runs from its root, with
the virtual environment active.

```
git clone https://github.com/Quentincls/localmesh-engine.git
cd localmesh-engine

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows, PowerShell
.venv\Scripts\activate.bat          # Windows, cmd.exe
source .venv/bin/activate           # Linux

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Run the one activation line that matches your shell, not all three.

The order matters because the three CUDA extensions of the next step are
compiled against this exact torch ABI. Build them against another torch and they
compile, then fail to load with a DLL or symbol error that names nothing useful.

Check:

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output: `2.8.0+cu128 12.8 True`.

## 3. The three CUDA extensions

`cumesh`, `o_voxel` and `flex_gemm` are three separate projects, all MIT, none
on PyPI. `o_voxel` is a subfolder of
[github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT,
Microsoft). `cumesh` and `flex_gemm` have repositories of their own,
[CuMesh](https://github.com/JeffreyXiang/CuMesh) and
[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) (MIT, Jianfeng Xiang); the
TRELLIS.2 setup script fetches them from there. The engine imports all three at
module level and cannot run without them.

**On Windows, Python 3.12, torch 2.8.0+cu128, you can skip this step.** The
four extensions are published prebuilt, under
[the `wheels-cp312-cu128` release](https://github.com/Quentincls/localmesh-engine/releases/tag/wheels-cp312-cu128):
download the four `.whl` and `pip install --no-deps` them, then go to step 4.
They are builds of other people's MIT code, taken unchanged from an environment
built by following this page. On any other Python, platform or CUDA, they will
not load and you build from source below.

Build them outside the repository, from the folder above it:

```
cd ..

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git
pip install --no-build-isolation ./CuMesh

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git
pip install --no-build-isolation ./FlexGEMM

git clone --recursive https://github.com/microsoft/TRELLIS.2.git
pip install --no-build-isolation ./TRELLIS.2/o-voxel

cd localmesh-engine
```

`--recursive` is not optional: `o-voxel` carries Eigen as a git submodule, and a
plain clone stops the build on a missing header. `--no-build-isolation` builds
against the torch you just installed instead of a fresh one pip fetches for the
build. On Windows, run all of this from the x64 Native Tools Command Prompt,
with the virtual environment activated, so the compiler is on PATH.

**Never run `pip install cumesh`.** It succeeds, and it installs a different
project that carries the same name (cumesh 0.1.0, Congjie He). You then get an
unreadable API error during generation instead of a clean failure.

Check:

```
python -c "import cumesh, flex_gemm; print('extensions ok')"
```

`o_voxel` will not import on its own, and that is expected: its `postprocess.py`
carries `import nvdiffrast.torch as dr` at module level. Do not install
nvdiffrast. Its licence forbids commercial use, the UV rasterisation is
rewritten in `uvraster.py`, and `leurre_nvdiffrast.py` registers a stand-in
under that name so the import passes. `o_voxel` gets exercised in step 8,
through the engine.

Third party Linux wheels for the same three extensions exist at
[huggingface.co/siraxe/TRELLIS.2-4B_cuda_12.8.r12.8_wheels](https://huggingface.co/siraxe/TRELLIS.2-4B_cuda_12.8.r12.8_wheels)
(cp312, linux_x86_64). They are not ours and we did not build them.

## 4. The package

Back in the root of the repository, `localmesh-engine/`:

```
pip install -e .
```

That installs the runtime dependencies and the `localmesh-engine` command. Your
torch stays as it is: the pin reads `torch==2.8.0`, which `2.8.0+cu128` already
satisfies.

Four of those dependencies are there for reasons the imports do not show, and
they were found by installing this repository into an empty environment rather
than by reading it. `zstandard` is how `o_voxel` reads its compressed volumes.
`triton` (`triton-windows` on Windows) carries the `grid_sample` kernels of
`flex_gemm`. And `kornia` and `timm` are required by the BiRefNet code that is
downloaded with its weights and executed under `trust_remote_code`: no static
analysis of this repository could have named them.

## 5. natten, for the four views only

```bash
pip install --no-build-isolation "natten>=0.21"
```

`--no-build-isolation` is not optional here. natten is a compiled CUDA
extension, and it compiles against whatever torch it finds. With build
isolation, pip creates a clean environment, pulls its own torch into it, and
builds against that one. On Linux you get a wheel linked to the wrong ABI; on
Windows the CUDA half of the build is skipped, the install SUCCEEDS, and the
neighborhood attention silently falls back to its CPU path. The four view path
then runs, slowly, without ever saying what happened.

Check what you got before going further:

```bash
python -c "import natten; print(natten.has_cuda())"
```

`True` or the four view path is not usable. If it prints `False`, uninstall,
make sure `torch==2.8.0+cu128` is the one in the environment, and build again
with `--no-build-isolation`.

`pip install -e ".[multiview]"` declares the same dependency but goes through
pip's resolver, so prefer the line above. Reference version on the machine
above: 0.21.6.

The single photo path runs without it. The four view path imports it while
sampling, and the check that runs before a job does not look for it, so a
missing natten surfaces mid generation.

That same check ignores `models/multivue/cameras/`, the model that reads the
shooting angle of each photograph. Without it the engine falls back to a
convention and infers which way round the two profiles are, which mirrors the
subject when the inference is wrong. Nothing fails, so install it with the rest:
see [docs/WEIGHTS.md](WEIGHTS.md). The result says which of the two happened.

## 6. LOCALMESH_ROOT

Weights and caches live outside the code, under one root you choose.
`LOCALMESH_ROOT` names that root: it is the folder that holds `models/`.
(`LUMENGEN_ROOT` is still read, for installations that predate the name.)

Windows PowerShell:

```
$env:LOCALMESH_ROOT = "D:\LocalMesh"        # this shell only
setx LOCALMESH_ROOT "D:\LocalMesh"          # every new shell
```

Linux:

```
export LOCALMESH_ROOT=/opt/localmesh        # add it to ~/.bashrc to keep it
```

Create the folder yourself. On first run the engine creates what it writes
underneath, including the Hugging Face cache at `<root>/models/hf`.

## 7. Put the weights in place

Every large model is placed by hand. Sizes, addresses, licences and the exact
tree are in [docs/WEIGHTS.md](WEIGHTS.md). Folder names are read literally:
`models/TRELLIS.2-4B` and `models/facebook/dinov3-vitl16-pretrain-lvd1689m` are
spelled that way in the code.

One piece is not placed by hand. The cutout model, `ZhengPeng7/BiRefNet_HR`, is
fetched from the Hub at the first generation when the cache is empty, and it
loads with `trust_remote_code=True`, so remote Python runs. A machine that must
stay offline needs that cache primed first.

## 8. Check that it works

```
python -m localmesh_engine photo.png --tier draft --seed 42 --to sortie/
```

The installed command is the same entry point: `localmesh-engine photo.png
--tier draft --seed 42 --to sortie/`. Progress goes to stderr, one line
per stage, and the shape stage ticks once per sampling step. The path of the
file goes to stdout, resolved to an absolute path.

```
carte : NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 Go
    0.0%  loading models
    5.0%  models ready
    7.0%  reading reference
    9.0%  cutting out the subject
   12.0%  generating shape
    ...   generating shape, once per sampling step, up to 45.0%
   47.0%  cleaning the mesh
   65.0%  texturing
   90.0%  finishing
   96.0%  cleaning the atlas
   98.0%  rendering the thumbnail
  100.0%  done
C:\work\sortie\model_42.glb
<faces> faces, <seconds> s, pic <peak> Go, graine 42
```

Two of those lines depend on which bake path the run takes, so the list can be a
line or two shorter. The last two are what success looks like: a
`model_<seed>.glb` in the output folder, then the face count, the wall clock,
the peak on the card and the seed.

How long to wait. The first run reads 8.1 GB of weights from disk before it
starts, and the first generation also downloads the cutout model. Measured on
the RTX 4060 Laptop with 8 GB, over six subjects: `standard` with four views
takes 6 min 30 to 8 min 30, `high` 9 min 20 to 13 min. Texture is about 60
percent of that time. `draft` is the shortest of the four tiers, and the check
above uses it: a single photo at `--tier draft` is the fastest run the engine
offers.

Then the four views, once step 5 and the multi-view weights are in place:

```
python -m localmesh_engine face.png --right d.png --left g.png --back b.png --to sortie/
```

All three sides are required. Two or three views are refused rather than served
under a name they were not measured under.

## 9. When it does not work

| Symptom | Cause | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'cumesh'` on any import of the engine | the three extensions are not built | step 3. `pip install cumesh` does not fix this, it installs a different project |
| An API error from `cumesh` that mentions no function you can find | the PyPI homonym is installed | `pip uninstall cumesh`, then build it from JeffreyXiang/CuMesh, step 3 |
| `ImportError: DLL load failed while importing ...` on Windows, or `undefined symbol: _ZN3c10...` on Linux | torch moved after the extensions were built | reinstall `torch==2.8.0` from the cu128 index, or rebuild the three extensions. Any `pip install` can upgrade torch as a side effect |
| `ERROR: ... requires a different Python` when installing the package | the interpreter is not 3.12 | Python 3.12 exactly. natten and the three extensions are cp312 builds |
| `ModuleNotFoundError: No module named 'nvdiffrast'` from a bare `import o_voxel` | o_voxel was imported outside the engine | import it through the engine, which registers the stand-in first. Do not install nvdiffrast |
| `LocalMesh Engine cannot find its runtime.` | `LOCALMESH_ROOT` is unset, or not visible to this process | step 6, then open a new shell so the variable is inherited |
| `TRELLIS.2 weights not found at <path>` while the files are on disk | the root is off by one level, or the folder is not named exactly `TRELLIS.2-4B` | the path in the message is the one the engine reads. That folder must hold `ckpts_fp8/` and a descriptor named `pipeline.json` or `pipeline_fp8.json`; both names are accepted |
| `IndexError: list index out of range` while loading models | the structure decoder is missing | put `ss_dec_conv3d_16l8_fp16.json` and `ss_dec_conv3d_16l8_fp16.safetensors` under `models/microsoft/TRELLIS-image-large/ckpts/`. With no local file the loader falls back to the Hub and splits an absolute path on `/`, which is where that IndexError comes from |
| An error naming `models/facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 is not in place, or access is not granted yet | step 0, then docs/WEIGHTS.md. Both paths need it |
| `The multi-view module is not installed. Missing: ...` | a four view weight file is absent AND the ported path was asked for by name (`chemin_mv="porte"`) | the message lists what is missing, file by file. docs/WEIGHTS.md. Unasked, missing weights do not raise: the run falls back to the older blend without saying so |
| `ModuleNotFoundError: No module named 'natten'` in the middle of a four view run | natten is not installed | step 5 |
| `torch.OutOfMemoryError: CUDA out of memory` | the tier is too high for this card, or the card is shared | drop to `--tier standard` or `--tier draft`, and close anything else holding video memory. A 3D viewer open during a run takes its share of the same 8 GB |
| `le palier max demande plus de mémoire que cette carte n'en a`, exit code 3 | `max` declares a 24 GB floor | use `draft`, `standard` or `high`. The check allows half a gigabyte of slack |
| `le multivue attend les TROIS côtés` | one side is missing | pass `--right`, `--left` and `--back` together |
| `No CUDA GPU visible.` | torch sees no card | check the driver, and check that `torch.cuda.is_available()` in step 2 returned `True` |

The four tiers, what each one costs and what it buys, are in
[docs/RECIPES.md](RECIPES.md). What the repository holds and what it is built
on: [README.md](../README.md).
