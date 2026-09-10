# Recipes, frozen

A tier is a contract, not a knob. The four recipes below are the ones frozen in
engine 1.0.0.

Every value on this page was read back from the code, and the code is named
next to it: `config.py` for the tiers, `pipeline.py` for the ceilings and the
fallback, `remaillage.py` for the cell budget, `matting.py` for the common
crop, `multivue/recette.json`, `multivue/forme.py` and `multivue/chemin.py`
for the four view path. Those paths are relative to `localmesh_engine/`, the
package folder of this repository. Change one of these values and you change
the contract, so you change the version of the engine.

## The four tiers

| | Draft | Standard | Detailed | Extreme |
|---|---|---|---|---|
| Identifier | `draft` | `standard` | `high` | `max` |
| Shape grid (`pipeline_type`) | 512 | 1024 | 1024 | 1536 cascade |
| Steps: structure, shape, texture | 8 | 12 | 16 | 16 |
| Sampler | Euler | Euler | Euler | Euler |
| Texture atlas | 2048 | 4096 | 4096 | 8192 |
| Face budget | 80,000 | 150,000 | 250,000 | 400,000 |
| Texture guidance | 1 | 5 | 5 | 5 |
| Memory required | none | none | none | 24 GB |

Source: the `PALIERS` table in `config.py`. The three step counts are equal
inside a tier, so one column holds them all: `draft` is 8 / 8 / 8, `standard`
12 / 12 / 12, `high` and `max` 16 / 16 / 16.

`standard` is the default, both in `config.PALIER_DEFAUT` and on the command
line, and these four identifiers are the only ones the command line accepts.
Extreme is the only one of the four that asks for memory, and the check allows
a card half a gigabyte short: `config.palier_tenable` compares `vram_gb + 0.5`
against the requirement, so a 23.5 GB card is served.

Extreme has not been measured on its target card. The comment above it in
`config.py` says so plainly: the machine that produced these numbers has 8 GB,
and the 24 GB figure is extrapolated from the cell law, not observed. Read
that one line as an estimate until someone runs it on a 24 GB card.

The dual contouring grid follows the shape grid: 512 for Draft, 1024 for
Standard and Detailed, 1536 for Extreme. It is not a field you set, it is
`Recette.dc_resolution`, a derived property, and it is the grid asked for, not
always the grid delivered: `remaillage.nettoyer` counts the cells the surface
crosses, brings that count back to a 1024 grid, and steps the grid down the
scale 1536, 1024, 768, 512 until it fits the cell budget of the card. On a very
open subject the mesh note says it happened. Three texture sampler settings
are shared by all four tiers, so they live in `remaillage.py` and not in the
tier table: guidance rescale 0.05, guidance interval [0.6, 0.9], rescale_t 3.0.

## What each tier actually buys

The scale is not linear, and saying so avoids false expectations.

**Draft to Standard.** The grid doubles, 512 to 1024, and so does the atlas,
2048 to 4096. Steps go from 8 to 12. This is the visible jump.

**Standard to Detailed.** The grid does not move. You buy steps and triangles,
not shape resolution. The difference reads on small parts, not on the
silhouette.

**Detailed to Extreme.** The grid goes to 1536: 1.5 times finer per axis, 3.4
times more cells. It is the only lever that adds geometric detail, and it is
why the tier asks for 24 GB.

The 8K atlas of Extreme is not there for sharpness. Measured on one subject,
same seed, same 8 GB card, at the Detailed tier:

| Atlas | Time | Peak VRAM | File | Texels per triangle |
|---|---|---|---|---|
| 4096 | 556 s | 10.3 GB | 56 MB | 68 |
| 8192 | 808 s | 11.4 GB | 162 MB | 271 |

Four times the texels, and no gain visible on screen: the model produces its
material at a fixed internal resolution, and a larger atlas is a larger canvas
for the same drawing. Extreme uses 8192 for packing instead: 400,000 faces on
a 4096 atlas leave 42 texels per triangle, under the threshold near 50 where
UV islands bleed into each other. On 8192 they get 168.

## Budgets computed from the card

Nothing is fixed in advance. Three ceilings follow from the memory found at
start up, and it is those ceilings that govern, not the requested resolution.

| Ceiling | Rule | On 8 GB | On 24 GB | Where |
|---|---|---|---|---|
| Tokens, fine stage | 2,600 per GB, floor 8,192 | 20,800 | 62,400 | `pipeline.py` |
| Faces accepted for remeshing | 1,000,000 per GB, floor 4,000,000 | 8,000,000 | 24,000,000 | `pipeline.py` |
| Remeshing cells | 1,000,000 per GB, no floor | 8,000,000 | 24,000,000 | `remaillage.py` |

Upstream TRELLIS.2 ships a token cap of 49,152, a datacenter figure. Ours only
bites from the 1536 cascade on: at 1024 the truncation loop never drops below
the requested resolution, and the cost of that cascade is intrinsic. Below
1536, the face and cell budgets are what govern.

The cell budget counts the CELLS the surface crosses, not the input triangles.
Measured 2 September 2026 over fifty two remeshes: the correlation between the
two is only +0.76, with clear inversions, and one subject of 23 million faces
cost half the memory of another at 6 million. When CUDA is not available, the
face ceiling and the cell budget both assume 8 GB rather than failing.

## Fallback rather than refusal

When the raw shape passes the face ceiling, the job does not stop. It restarts
one tier down, with the same resolved seed and the full recipe of that lower
tier: steps, atlas, sampler and texture guidance are reset so they come from
the lower recipe. A face budget the caller set explicitly is kept, and the
mesh note says what happened. The map is `pipeline.PALIER_DE_REPLI`:
`standard` falls back to `draft`, `high` to `standard`, `max` to `high`.

A shape too heavy almost always comes from four views above Draft, and the
same request one notch down goes through. Bench of 4 September 2026: 5.9 M
faces at Standard, 1.6 M at Draft. `LUMENGEN_SANS_REPLI` turns the fallback
off; the job then fails with the face count and the ceiling in one line.

The ported four view route does not fall back, it decimates.
`multivue/budget.reduire_si_trop_lourd` brings the raw shape down to 97 percent
of that same face ceiling and the mesh note says so. Falling back there would
throw away four encodings, one fusion and two cascades for an overrun that is
often a few percent: the traffic light comes out at 8,234,430 faces against a
ceiling of 7,995,605. The tier fallback therefore fires on the single photo
route and on the legacy blender.

## Four views

The four view path fires when the request carries, besides the front photo,
the three sides named `droite`, `gauche` and `dos`. There is nothing to
enable, and two things to have installed: the four view weights under
`<root>/models/multivue`, and `natten`. Without the weights the job still
succeeds, on the legacy blender, with no warning: `_multivue_porte` returns
false and the other branch takes it. Without `natten` the ported path stops at
the encoding step on an import error.

It is measured at four views, not two or three: the command line refuses
incomplete sets rather than returning an object whose value nobody knows.

### Which tiers it serves, and why not the top one

| Tier | Shape resolution promised | Structure steps, shape steps per cascade stage | Path |
|---|---|---|---|
| Draft | 512 | 8, then 4 per stage | ported path, returns 1024, more than promised |
| Standard | 1024 | 12, then 6 per stage | ported path |
| Detailed | 1024 | 16, then 8 per stage | ported path |
| Extreme | 1536 | not applicable | legacy blender |

Read the step column carefully. `pipeline.py` passes `steps_structure` whole
and `max(1, steps_shape // 2)` for the shape, and `multivue/forme.py` applies
that shape count TWICE, once per cascade stage, 512 then 1024. Standard
therefore runs 12 structure steps and 6 + 6 = 12 shape steps, not 6.

The multi-view weights exist at 512 and 1024 only: `forme_512_mv_fp8` and
`forme_1024_mv_fp8`, alongside `structure_mv_fp8`. Above 1024 the ported path
cannot keep the promise of the tier, so every tier on the 1536 shape grid goes
back to the legacy blender, which does follow the tier. The alternative is to
ship a Standard shape under another label. `LUMENGEN_MV_TOUS_PALIERS=1` lifts
the barrier, for measurement.

The rest of `multivue/recette.json` is shared by every tier on this path:
guidance strength 7.5 at both stages, guidance interval [0.6, 1.0], guidance
rescale 0.7 for structure and 0.5 for shape, rescale_t 5.0 and 3.0. One frozen
value of this path lives in `multivue/structure.py` rather than in the recipe
file: the force that decides how much each view weighs on a cell.
`force_selon_desaccord` returns 0, so the four views are averaged flat unless a
bench sets `LUMENGEN_MV_VISIBILITE` or `force_mv` on the request. Measured
9 September 2026, eleven generations, one variable at a time: on four runs
across two subjects and two tiers, force 0 left fewer sharply folded edges and
far fewer pieces than force 3, and the doubled features the force had been
written for did not come back.

The legacy blender has one value of its own, `blend_temperature`, default 4.0 in
`pipeline.py`: how sharply the four views split a voxel between them. Measured
on a bench subject, 4 September 2026, 4.0 against the 2.0 inherited from the
ComfyUI node gives sharper detail for 5 % more time.

### The four views share one crop

A single photo is cropped on the largest dimension of the subject. On an
object taller than it is wide from every angle, that is the height: the four
crops already agree, and batching them changes nothing, to the pixel. On an
elongated object it is not. A motorcycle seen from the side is cropped on its
LENGTH, so its height fills 65 % of the frame against 96 % from the front.
Measured 10 September 2026 and recorded in `matting.py`: 37.9 % of scale gap on
the motorcycle, 39.7 % on a cassette, 0.0 % on a samurai and on a head.

The rule that settles it: a rotation around the vertical axis does not change
the height of an object, so height alone says whether two views are at the
same scale. The engine takes a common crop side proportional to the subject
height in each view, with the smallest factor that still fits the widest view.
What it costs: on an elongated object the subject fills 64 % of the frame
instead of 96 %, and the models were trained on subjects that fill it. That is
cheaper than the scale disagreement it replaces. `cadre_mv="separe"` on the
request puts the per view crop back, for the bench.

## What the engine writes next to the weights

At load time the engine reads `pipeline.json` or `pipeline_fp8.json` from the
weights folder and writes a side car named `pipeline.lumengen.json` beside it.
Exactly one field is rewritten: the matting model, from `briaai/RMBG-2.0`,
gated and non commercial, to `ZhengPeng7/BiRefNet_HR`, which is MIT. Checkpoint
paths stay relative. That file name carries the historic project name, and it
is part of the contract until a release changes it.

## Progress labels

The strings the engine passes to the progress callback are part of the
contract: integrators match them exactly to translate or to drive a bar, so
renaming one is a version change. The main route, in order: `loading models`,
`models ready` (both from the model load, rescaled into the first 5 percent of
the bar), `reading reference`, `cutting out the subject`, `checking the two
profiles` (four views only), `generating shape`, `cleaning the mesh`,
`texturing`, `baking PBR materials`, `finishing`, `cleaning the atlas`,
`rendering the thumbnail`, `done`. Two stages add sub labels of their own: the
ported four view path, in `multivue/chemin.py`, sends `encoding views`,
`merging views`, `encoding views` again for the fine stage, then `generating
shape`; remeshing, in `remaillage.py`, sends `filling holes`, `remeshing`,
`removing debris`, `decimating`, `closing cavities`.

## Bench overrides

These environment variables move a recipe. They exist so a measurement can
change one thing at a time, and nothing sets them in normal use. They still
carry the historic `LUMENGEN_` prefix. The runtime root is the exception:
`config.py` reads `LOCALMESH_ROOT` first and falls back to `LUMENGEN_ROOT` for
installations that predate the name.

| Variable | Default | Effect |
|---|---|---|
| `LUMENGEN_MV_TOUS_PALIERS` | unset | `=1` opens the ported path on the 1536 shape grid |
| `LUMENGEN_MV_ANCIEN` | unset | `=1` forces the legacy blender |
| `LUMENGEN_SANS_MULTIVUE_PORTE` | unset | set: disables the ported path |
| `LUMENGEN_MV_PAS` | 6 | shape steps per cascade stage when the caller passes none |
| `LUMENGEN_PLAFOND_FACES` | per card | overrides the face ceiling |
| `LUMENGEN_SANS_REPLI` | unset | set: refuse instead of falling back one tier |
| `LUMENGEN_BANC_MELANGE` | 4.0 | blend temperature of the legacy blender |
| `LUMENGEN_TRANSPORT_CADRE` | unset | `=1` reprojects the views into the original frame |
| `LUMENGEN_MV_VISIBILITE` | unset, force 0 | any value turns the view weighting on, at that force |
| `LUMENGEN_MV_REGLE` | unset, anisotropic position | `visibilite` or `normale` picks another weighting rule |
| `LUMENGEN_ECART_VUES_MAX` | 6.0 | percent of height gap above which the result warns |
| `LUMENGEN_TRACE_MAILLAGE` | unset | `=1` writes `avant_texture.glb`, the mesh as it enters texturing |
| `LUMENGEN_FIX_TILED_MESH` | 1 | `=0` restores the upstream tile extractor, without the one owner per quad rule |

Four more knobs live on the request itself, so a bench can change them
without restarting the engine: `chemin_mv` (`auto`, `porte`, `ancien`),
`angles_mv` (`auto`, `nominaux`), `cadre_mv` (`auto`, `separe`) and `force_mv`
(`None` for the measured rule, a number to impose the view weighting force).

## Changing a recipe

Three conditions, together:

1. the bench table, before and after, on the reference subjects
   ([CONTRIBUTING.md](../CONTRIBUTING.md) gives the command, the columns and
   the seed rule; the subject folder ships empty, you bring your own images);
2. the peak memory measurement on an 8 GB card, because that is the target;
3. the update of this file, in the same commit as the change.

A recipe that changes without this file is an invisible change for everyone
building on top of it.
