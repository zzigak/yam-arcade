# SAM 3D assets (yam-arcade)

Made with SAM 3D Objects (`~/tools/sam-3d-objects`) from single real photos: photo -> object mask (rembg isnet, or
SAM-ViT-H box prompt where several objects touched) -> SAM 3D (2 seeds for the important objects, best kept) ->
post-processing in `_work/scripts/process.py` (all scripts, masks, overlays and raw outputs are under `_work/`).

## Conventions (read this)

- **Frame: Z-up, metres, base at z = 0, bounding box centred on x = y = 0.** Objects with a handle are yawed so the
  **handle points to +x** (the pitcher's spout therefore points to -x). `info.json -> collision.cavity_center_xy_m` is
  the centre of the opening (it is not at x = y = 0 when there is a handle).
- **OBJ files are Z-up as-is** (Genesis `gs.morphs.Mesh(file=...obj)` with its default `file_meshes_are_zup=True`
  loads them upright; in Blender import with `up_axis='Z', forward_axis='Y'`).
- **`visual.glb` follows the glTF spec (Y-up)**: Blender / three.js import it upright. If you load it with trimesh,
  or in Genesis, you get raw Y-up coordinates: map `(x, y, z) -> (x, -z, y)` or pass `file_meshes_are_zup=False`, or
  just use `visual.obj`.
- Colours are **per-vertex** (SAM 3D gives no UV texture), in `visual.glb` (COLOR_0) and as `v x y z r g b` in the OBJs.
- `visual.obj` ~200k faces, `visual_lo.obj` 40k faces (decimated, colours re-sampled); neither is guaranteed
  watertight. Use `collision_hollow.obj` for physics.
- **`collision_hollow.obj`** (containers): watertight, open-topped, solid bottom, solid handle, walls >= the stated
  thickness, inner cavity following the SAM 3D cavity (or, for the pitcher and urn, the outer shape offset inward),
  a few thousand faces. Built by voxelising the scaled mesh (0.6-1.5 mm voxels), thickening thin walls to the target,
  removing lids, marching cubes, light Taubin smoothing and quadric decimation. The rim is slightly ragged
  (+-1 voxel). Lids (pitcher, jar, urn) exist only in the visual; the collision mesh is open at the top.
- **Capacity** = brimful volume of the collision cavity with the object upright (the level at which water would
  spill over the lowest rim point), from the voxel grid.
- Masses are not set here (choose them per station: e.g. glass 2.5 g/cm^3, porcelain 2.4, PP/SAN plastic ~1.1).

## Assets

| name | bbox x * y * z (cm) | capacity (L) | collision_hollow | verdict | seed | source photo - author - license | notes |
|---|---|---|---|---|---|---|---|
| `pitcher` | 26.1 x 13.5 x 25.0 | 3.50 | yes (8000 f, wall 4.0 mm, floor top 4 mm) | good | seed 42 | [File:Brita water pitcher.png](https://commons.wikimedia.org/wiki/File:Brita_water_pitcher.png) - Bas - CC BY 4.0 | Brita-style filter pitcher (clear plastic in the photo; SAM 3D colours it grey-white). Visual has the lid; collision is open-topped (lid removed above 90% height), walls 4 mm, spout at -x, handle +x. Depth set from Brita proportions (see Size choices). A string tied round the handle in the photo was trimmed and painted grey; a ~1.5 cm frayed bump remains under the handle's lower arm. The visual's bottom ~3 cm is solid (SAM 3D read the water in the photo as solid); the collision cavity ignores it. Seed 42 kept over seed 7 (seed 7's lid was a flat disc). |
| `mug` | 12.9 x 9.2 x 9.5 | 0.42 | yes (4000 f, wall 3.0 mm, floor top 3 mm) | very good | seed 42 | [Black coffee mug white table](https://www.rawpixel.com/image/6038747/photo-image-public-domain-coffee-table) - n/a - CC0 1.0 | dark grey bistro mug, white glaze inside, open handle. |
| `tumbler` | 9.4 x 9.4 x 14.0 | 0.55 | yes (4000 f, wall 3.0 mm, floor top 8 mm) | very good | seed 42 | [Cracked glass](https://www.flickr.com/photos/61904084@N00/7175035520) - thedailyenglishshow - BY 2.0 | straight conical pint-style glass (photo shows a faint crack line in the texture). |
| `tumbler_curvy` | 8.4 x 8.6 x 14.0 | 0.49 | yes (4000 f, wall 3.0 mm, floor top 8 mm) | good | seed 42 | [Empty Pint](https://www.flickr.com/photos/41894148532@N01/4291434) - Neil T - BY-SA 2.0 | alternate tall glass with a Coca-Cola-style waist; some smudges baked into the colours. |
| `teacup` | 11.2 x 9.4 x 6.5 | 0.28 | yes (4000 f, wall 3.0 mm, floor top 3 mm) | very good | seed 42 | ['Bamboo' Motif Teacup Saucer Christoph](https://www.rawpixel.com/image/11797921/bamboo-motif-teacup-and-saucer-christopher-dresser-and-mintons-ltd) - n/a - CC0 1.0 | wide low porcelain teacup (Dresser/Minton bamboo pattern), angular handle; saucer not included. |
| `wineglass` | 7.0 x 6.9 x 15.0 | 0.09 | yes (4000 f, wall 3.0 mm, floor top 75 mm) | good | seed 7 | [Glass](https://www.flickr.com/photos/55247565@N05/6058691104) - olga.reznik - BY 2.0 | small stemmed glass (sherry / small wine); bowl ~5.5 cm across, floor of the bowl at 7.7 cm, solid stem and foot. Glass is rendered opaque grey. |
| `mug_white` | 10.8 x 7.6 x 9.5 | 0.36 | yes (4000 f, wall 3.0 mm, floor top 4 mm) | ok (alternate) | seed 42 | [POTD 12/1/2008](https://www.flickr.com/photos/66526627@N00/2189512452) - ptc24 - BY 2.0 | white classic mug; the photo had tea in it, so the visual has a tea-coloured surface part way down; collision cavity is clean. |
| `pitcher_glass` | 22.2 x 17.7 x 18.0 | 2.67 | yes (6000 f, wall 3.0 mm, floor top 8 mm) | good (alternate pourer) | seed 42 | [Water Pitcher LACMA M.2004.176.1](https://www.flickr.com/photos/50398299@N08/15829809684) - Fæ - PDM 1.0 | round glass water jug with a pouring lip at -x and an open top (no lid); museum piece (LACMA), glass rendered opaque grey. Handle +x. |
| `urn` | 24.4 x 26.2 x 30.0 | 8.54 | yes (8000 f, wall 4.0 mm, floor top 4 mm) | good (optional) | seed 42 | [Acrylic Beverage Dispenser](https://www.flickr.com/photos/38227433@N04/17247818641) - FestivitiesMN - BY-NC-SA 2.0 | acrylic drink dispenser with a tap facing -y; visual has its lid, collision is open-topped (lid cut at 86% height). Tap is visual only (solid). |
| `jar` | 7.2 x 7.2 x 10.0 | 0.24 | yes (4000 f, wall 3.0 mm, floor top 3 mm) | good (reference) | seed 42 | [jars](https://www.flickr.com/photos/87481332@N00/140231113) - How can I recycle this - BY 2.0 | hexagonal glass jam jar with a metal twist-off lid (visual). Collision: jar body only, lid region (top 14%) removed, no threads, wall locally ~1.5 mm at the cut neck - the jars station builds real threads procedurally; use this for proportions/look. |
| `jar_lid` | 6.8 x 7.0 x 1.2 | - | - | good (reference) | seed 42 | [Bento #293: Mason jar bento - Mexican](https://www.flickr.com/photos/25977089@N00/7150219611) - goblinbox_(queen_of_ad_hoc_bento) - BY 2.0 | metal mason-jar lid lying open side UP (as photographed); flip 180 deg about x to cap a jar. 1.2 cm tall, visual only. |
| `mahjong_tile` | 3.0 x 2.3 x 1.4 | - | - | very good (reference) | seed 42 | [File:Mahjong Tiles (195606345).jpeg](https://commons.wikimedia.org/wiki/File:Mahjong_Tiles_(195606345).jpeg) - Flickr user via Commons (see page) - CC0 | one tile face up (+z) showing the 4-circles face, blue back layer at the bottom; long edge along x. Visual only (use a box collider). |
| `box` | 32.0 x 26.1 x 15.7 | 6.96 | yes (3000 f, wall 4.0 mm, floor top 4 mm) | good | seed 42 | [Empty Box](https://www.flickr.com/photos/34316169@N08/3189567558) - z287marc - BY 2.0 | open-top cardboard box (flaps removed), long side along x. Collision = the visual's convex hull carved to a 4 mm-wall open box (SAM 3D's own cardboard walls were too thin and ragged to voxelise cleanly); its rim follows the hull top, which slopes 1.5 cm from corner to corner. Rescale freely if the cables station needs another size. |

## Pouring quick facts (station frame of each asset)

| name | opening centre x, y (cm) | brim level z (cm) | floor top z (cm) | spout tip x, y, z (cm) | handle |
|---|---|---|---|---|---|
| `pitcher` | -1.2, 0.1 | 24.5 | 0.4 | -13.1, 0.0, 24.2 | +x |
| `mug` | -1.9, -0.0 | 9.5 | 0.3 | - | +x |
| `tumbler` | -0.0, -0.1 | 14.0 | 0.8 | - | - |
| `tumbler_curvy` | 0.0, 0.0 | 14.0 | 0.8 | - | - |
| `teacup` | -1.1, 0.0 | 6.4 | 0.3 | - | +x |
| `wineglass` | 0.0, 0.0 | 15.0 | 7.5 | - | - |
| `mug_white` | -1.4, -0.0 | 9.5 | 0.4 | - | +x |
| `pitcher_glass` | -2.0, -0.0 | 17.5 | 0.8 | -9.2, 0.2, 17.3 | +x |
| `urn` | 0.0, 1.7 | 27.0 | 0.4 | - | - |
| `jar` | 0.0, -0.0 | 10.0 | 0.3 | - | - |
| `box` | -0.1, -0.1 | 13.5 | 0.4 | - | - |

## Not delivered / rejected

- `tile_rack`: no usable photo found (Commons has only a diagram, `File:Mahjong-rekje.png`; Openverse hits were tiles, not racks). It is a simple shape - build it procedurally (e.g. a ~45 x 5 x 3 cm bar with a front lip and a 10-15 deg backward tilt).

## Status

Complete. A re-run of the pitcher with a cleaner mask (string removed, handle hole kept open: `_work/inputs/pitcher_brita2*`) is prepared but did not run: the shared GPU never dropped below 5 GB in 15 min. To redo it: `_work/scripts/run_sam3d.sh _work/scripts/jobs3.json _work/sam3d_raw`, then `process.py pitcher:42` (genesis venv), `qa.sh pitcher`, and point `selection.json` at it.

## Size choices

- `pitcher`: height = 25.0 cm - Brita 6-10 cup pitchers are 25-26 cm tall (e.g. Everyday 10-cup 26.2 cm); 25 cm chosen; depth rescaled x0.7774: photo is a pure side view, so SAM 3D had to guess the depth (17.5 cm, a round body); Brita 10-cup pitchers are 26.9 x 14.0 x 26.2 cm (L x W x H), so y is rescaled to 13.5 cm
- `mug`: height = 9.5 cm - standard 11-12 oz coffee mug, 9.5 cm tall (task spec)
- `tumbler`: height = 14.0 cm - conical drinking glass, 14 cm tall
- `tumbler_curvy`: height = 14.0 cm - tall drinking glass (pint/highball class), 14 cm tall (task spec)
- `teacup`: height = 6.5 cm - porcelain teacup: 6-7 cm tall, ~9 cm rim; 6.5 cm chosen
- `wineglass`: height = 15.0 cm - small stemmed (sherry / small wine) glass, 15 cm tall
- `mug_white`: height = 9.5 cm - standard coffee mug, 9.5 cm tall
- `pitcher_glass`: height = 18.0 cm - bulbous glass water jug; 18 cm tall keeps the brimful capacity near 2 L
- `urn`: height = 30.0 cm - ~2 gallon (7.5 L) acrylic beverage dispenser, ~30 cm tall without a stand
- `jar`: height = 10.0 cm - hexagonal jam jar (~280 ml) with a 63-70 mm twist-off lid, ~10 cm tall
- `jar_lid`: length = 7.0 cm - regular-mouth mason jar band/lid: 70 mm outer diameter
- `mahjong_tile`: max = 3.0 cm - Chinese mahjong tile ~30 x 22 x 16 mm; largest dimension set to 30 mm
- `box`: length = 32.0 cm - open cardboard box sized for a 1.2 x 0.7 m table: 32 cm long (photo proportions give the rest)

## Licenses

Photos are CC0 / public domain / CC BY / CC BY-SA, except `urn` (CC BY-NC-SA 2.0: non-commercial research use only). Attribute the authors above if these meshes are shown publicly.
