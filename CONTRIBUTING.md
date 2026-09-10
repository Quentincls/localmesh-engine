# Contributing to LocalMesh Engine

This engine is judged on what it produces. The rules below are short, and they
do not bend.

## The bench decides

Nothing lands without a measurement. A change that claims fidelity, speed or
memory arrives with the bench table, before and after, on the same subjects and
the same seeds. A screenshot is not a measurement, and one object is not a
proof.

```
python -m localmesh_engine.banc.banc --vers resultats/
python -m localmesh_engine.banc.banc --palier high --seulement samourai_mv --vers resultats/
```

Run it as a module, from the repository root, or from anywhere once the
package is installed. The file imports the pipeline relatively, so
`python localmesh_engine/banc/banc.py` fails on the import.

`--palier` takes `draft`, `standard`, `high` or `max`, and defaults to
`standard`. `--seulement` runs a single subject by name. `--sujets` defaults to
`localmesh_engine/banc/sujets` and only needs to be passed to point somewhere
else. `--vers` receives one folder per subject and a `banc.json` rewritten after
every subject. The Markdown table goes to standard output at the end and is
written nowhere, so redirect it if you want to keep it.

### You bring the subjects

`localmesh_engine/banc/sujets/` carries nothing but its `LISEZ-MOI.md` in this
repository, and it stays that way. Reference photos belong to their authors, and
a code repository is not an image store. The `.gitignore` enforces it.

`localmesh_engine/banc/banc.py` names the files it expects and freezes one seed
per subject. Getting the engine to run at all is the long part: see
[docs/INSTALL.md](docs/INSTALL.md). The four view subjects also need the
`multiview` extra, which pulls `natten`.

Put your own images under those names, or add an entry to `SUJETS` with its own
fixed seed. Two measurements that do not share a seed do not compare. Publish
your images next to your table: a number nobody can reproduce is not a
measurement.

A subject added to the bench has to catch a defect the others do not show.

### The table it prints

| Column | What it reports |
|---|---|
| `sujet` | The name of the entry in `SUJETS`, so a row can be traced back to its images and its seed |
| `faces` | Faces delivered. A budget is a ceiling, not a target: two meshes at 150,000 faces do not carry the same surface |
| `duree_s` | Wall clock from the request to the file, not one chosen stage |
| `pic_vram_go` | What the run took on the card: peak card occupancy above its level at the start, sampled while the run goes, never below what PyTorch reserved. Allocations made outside PyTorch are counted |
| `bords_ouverts` | Open edges, counted after welding vertices by position |
| `morceaux` | Connected pieces |
| `eclats_moins_de_50_faces` | Pieces under 50 faces |
| `non_manifold` | Edges shared by more than two faces |
| `double_face` | The double sided flag written in the material |
| `metal_moyen` | Mean metallic channel of the delivered metallic roughness texture |

UV unwrapping duplicates vertices along every seam, so on raw indices a
perfectly closed object reports up to 39 percent boundary edges.

Pieces matter as much as open edges. A mesh can be watertight in topology and
still carry hundreds of detached fragments under its surface, which is why the
bench counts pieces and fragments under 50 faces separately.

## What is not acceptable

**A setting tuned for one subject.** The engine does not know the name of what
it is given. The multi-view mixer is the standing example: three weighting
rules live in `localmesh_engine/multivue/structure.py`, and
`localmesh_engine/multivue/forme.py` picks between them on
`LUMENGEN_MV_REGLE`, for the bench only. The choice bears on the shape
cascade alone: the structure stage keeps the positional proxy either way.
Measured on the back of the traffic light, likeness to the photo, same seed:
flat average 0.476, anisotropic position 0.429, depth visibility at force 6
0.456, isotropic position 0.377, estimated normal at force 3 0.298. The
retained rule is the anisotropic position proxy, which is not the best line of
that column. The depth test scores above it on that subject and costs on
another: samurai agreement falls from 0.711 to 0.627. One subject decides
nothing.

And the force that makes any of those rules bite is off in normal use. It is
`LUMENGEN_MV_VISIBILITE`, unset by default, and unset means force 0: the four
views are averaged flat, whichever rule is selected. Eleven generations on
9 September 2026, one variable at a time, put it there;
[docs/RECIPES.md](docs/RECIPES.md) records what force 3 cost.

**Speed paid in quality**, unless the trade is stated and measured.

**A recipe changed in silence.** The four tiers are frozen in
[docs/RECIPES.md](docs/RECIPES.md). Changing one changes the version of the
engine, not an implementation detail.

**One more dependency** without a reason that fits in one sentence. `torch` and
`torchvision` are pinned exactly in `pyproject.toml`, because the three CUDA
extensions are compiled against that ABI: a package that drags another torch in
breaks the install. A dependency that fetches something at run time has to be
said out loud: today the engine fetches exactly one piece by itself, the
`ZhengPeng7/BiRefNet_HR` matting model, on the first generation, and loads it
with `trust_remote_code=True`.

## The shape of a change

One fix per commit, with its message in plain words: what was wrong, what
changes, and the number that proves it. A message that says what was refactored
teaches nothing to whoever reads the log in six months.

Comments explain WHY, with the measurements that led there. That is the
convention of the whole repository, and it is what stops the same path from
being reopened ten times.

The code and its comments are in French. The documents are in English. A
contribution can be in either language.

The stage labels passed to the progress callback are part of the API. They are
the strings given to `step()` in `localmesh_engine/pipeline.py`, callers match
them exactly, so renaming one is a breaking change. The full list is in
[docs/RECIPES.md](docs/RECIPES.md).

Every contribution is offered under Apache-2.0, per section 5 of that licence.

## What a proposal must contain to be read

1. One sentence saying what changes.
2. The bench table, before and after, same subjects, same seeds, same tier.
3. The images you measured on, so the table can be reproduced.
4. Peak card memory on an 8 GB card.
5. [docs/RECIPES.md](docs/RECIPES.md) updated in the same commit, if a tier moves.

Without these, it is read as an idea, not as a proposal.

Proposals go to `github.com/Quentincls/localmesh-engine` as pull requests
against `main`. There is no test suite in this repository: the bench is the
check, and it needs a card.

## Memory is the real constraint

This engine exists to run on an 8 GB card. A proposal that assumes 24 GB is
refused by default, even when it is better: it takes the engine out of its
reason to exist.

Budgets are computed from the card. The token ceiling is 2,600 per gigabyte, so
20,800 on an 8 GB card against the 49,152 the vendored default declares. The
face budget is a second ceiling: on the four view route a shape above it is
decimated and the result says so in its notes, rather than falling back a tier:
at that point four views have already been encoded and fused, and falling back
would throw all of it away.

## What this repository covers

Generation: one photo, or four sides of the same subject. Gaussian splats, the
image workshop, quad retopology and the project library live in the application
that uses the engine, and their code is not here. The base model is TRELLIS.2:
a proposal that replaces it is a different engine, not a change to this one.
