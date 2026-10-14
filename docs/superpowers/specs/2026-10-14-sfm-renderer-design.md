# ScenePreview 3.0 — render a mesh or the SfM landmarks from every view

## Goal

Rewrite the `ScenePreview` Meshroom node so that it:

- takes an sfmData file (any format pyalicevision reads: `.sfm`, `.json`, `.abc`) and an optional mesh (`.obj`);
- renders, with Blender, the mesh — or the sfmData landmarks when no mesh is given — from the point of view of every reconstructed view;
- matches each view's pinhole intrinsics (focal, principal point, resolution, pixel ratio) and extrinsics exactly; lens distortion is ignored, so renders align with AliceVision *undistorted* images;
- writes one image per view to the node cache folder, exposed as an output with `semantic="image"` so the 2D viewer can browse it.

Rendering style stays the old preview style, chosen by node parameters: mesh as wireframe or line art, points as colored particles, optional undistorted image as background, optional masks.

## Constraints

- pyalicevision **cannot** be imported inside Blender's Python. Verified with blender 4.5.3 and aliceVision 3.3.7: (1) Blender links Python statically, and pyalicevision needs `libpython3.11.so.1.0`; (2) with python-3.11.14's libpython added to `LD_LIBRARY_PATH`, the import then fails on `libaliceVision_image.so: undefined symbol OpenImageIO_v3_0::ustring::empty_std_string`, because Blender bundles its own, incompatible OpenImageIO. AliceVision code and `bpy` code must live in separate processes.
- Packages come from rez. `package.py` already declares `MRBLENDERSFMRENDERER_SUBREQUIRES = blender-4:aliceVision`, so the node process has pyalicevision and `blender` on `PATH`. No change to `package.py`.
- The node replaces `ScenePreview` (same class name), version bumped `2.0` → `3.0`.

## Architecture

Two stages per chunk:

```
Meshroom process (pyalicevision)            Blender subprocess (bpy only)
ScenePreview.processChunk                   scripts/render.py
  load sfmData                                read chunk JSON
  select views of the chunk range    ──►      build scene, import mesh / landmarks PLY
  write chunk_<i>.json (+ landmarks.ply)      for each view: set camera, compositing, render
  run blender -b --python render.py -- json   exit code != 0 on fatal error
```

### Files

| File | Role |
|---|---|
| `meshroom/blenderSfmRenderer/ScenePreview.py` | Node description + `processChunk`. Imports pyalicevision. |
| `meshroom/blenderSfmRenderer/scripts/render.py` | Blender script. Imports `bpy` and `cameraMath` only. Replaces `preview.py`. |
| `meshroom/blenderSfmRenderer/scripts/cameraMath.py` | Pure Python (no `bpy`, no pyalicevision): converts a pinhole camera to Blender camera parameters and `matrix_world` as nested lists. Unit-testable anywhere. |
| `meshroom/blenderSfmRenderer/__init__.py`, `scripts/__init__.py` | Restored from HEAD (currently deleted in the working tree). |
| `tests/test_cameraMath.py` | Unit tests (pytest). |

`preview.py` is deleted; its shading and compositing functions move to `render.py` mostly unchanged.

## Node interface

Inputs (names kept from 2.0 where meaning is unchanged):

| Name | Type | Notes |
|---|---|---|
| `blenderCmd` | File | Default `blender`, `invalidate=False`. |
| `cameras` | File | sfmData, any pyalicevision-readable format. |
| `model` | File | Optional `.obj` mesh. Empty → render sfmData landmarks. A `.abc` value (from 2.0 graphs) logs a warning and falls back to landmarks. |
| `useBackground`, `undistortedImages` | Bool, File | Unchanged. |
| `useMasks`, `masks` | Bool, File | Unchanged (empty `masks` → background alpha, as in 2.0). |
| `pointCloudParams` (`particleSize`, `particleColor`) | Group | Enabled when `model` is empty or not `.obj`. |
| `meshParams` (`shading`, `edgeColor`) | Group | Enabled when `model` ends with `.obj`. |

The `script` input is removed (the node knows its script path).

Outputs: unchanged from 2.0 — `output` = `{nodeCacheFolder}`, `frames` = `{nodeCacheFolder}/<FILESTEM>_preview.jpg` with `semantic="image"`.

Parallelization: unchanged — `size = avpar.DynamicViewsSize("cameras")`, `Parallelization(blockSize=40)`.

## Stage 1 — `processChunk` (pyalicevision)

1. `sfmDataIO.load` the `cameras` file (views, intrinsics, extrinsics, plus structure only if landmarks are needed).
2. Take views sorted by `viewId`, slice `[chunk.range.start : chunk.range.end]`.
3. Keep views where `sfm.isPoseAndIntrinsicDefined(view)`; log skipped ones.
4. For each kept view, read from the `IntrinsicScaleOffset`: width, height, scale `(fx, fy)` in pixels, offset `(ox, oy)` = principal point minus image center, in pixels; from the pose: rotation `R` (world→camera) and center `C`.
5. Write `{nodeCacheFolder}/chunk_<i>.json`:
   ```json
   {
     "settings": { "model": "...", "useBackground": true, "undistortedImages": "...",
                   "useMasks": false, "masks": "", "particleSize": 0.01, "particleColor": "Red",
                   "shading": "wireframe", "edgeColor": "Red", "landmarks": "<ply path or null>" },
     "views": [ { "viewId": "123", "outputPath": ".../<stem>_preview.jpg",
                  "width": 4000, "height": 3000, "fx": 3200.0, "fy": 3200.0,
                  "ox": 12.3, "oy": -4.5, "R": [[...],[...],[...]], "C": [x, y, z] } ]
   }
   ```
6. Landmark mode: write `{nodeCacheFolder}/landmarks_<i>.ply` (ASCII or binary little-endian, `x y z` float + `red green blue` uchar). Per-chunk file name avoids races between parallel chunks.
7. `subprocess.run([blenderCmd, "-b", "--python", render.py, "--", jsonPath])`, stream output to the chunk log; raise if the return code is non-zero.

## Stage 2 — `render.py` (bpy)

Same scene setup as 2.0 (Cycles, 1 sample, no denoise, film transparent, compositing graph for background/mask).

- **Mesh**: `bpy.ops.wm.obj_import(filepath, forward_axis='Y', up_axis='Z')` — keeps AliceVision world coordinates unchanged. Wireframe or line-art shading as in 2.0.
- **Landmarks**: `bpy.ops.wm.ply_import(..., forward_axis='Y', up_axis='Z')`, then the 2.0 geometry-nodes particle setup.
- **World convention**: everything stays in AliceVision world coordinates; only the camera's local axes are converted.
- Per view: apply `cameraMath` output, set background/mask image (same viewId/stem glob lookup as 2.0; skip the view if a required image is missing), render to `outputPath`, rescale if pixel ratio ≠ 1 (as 2.0), free images.

## Camera math (`cameraMath.py`)

AliceVision camera frame: x right, y down, z forward. Blender camera frame: x right, y up, looking along −z.

- `matrix_world = [[Rᵀ, C], [0, 1]] · diag(1, −1, −1, 1)`
- `sensor_fit = 'HORIZONTAL'`, `sensor_width = width` (pixel units), `lens = fx`
- `resolution = (width, height)`; pixel ratio `fy / fx` handled with `pixel_aspect_x` + post-rescale as in 2.0
- `shift_x = −ox / max(width, height)`, `shift_y = oy / max(width, height)` (Blender normalizes shift by the larger dimension; signs as in 2.0)

Correctness is defined by the projection test below, not by the formulas: if a sign is wrong the test fails and the formula is fixed.

## Error handling

- sfmData fails to load → exception, chunk fails.
- View without pose or intrinsic → skipped, logged.
- Non-pinhole intrinsic (equidistant, equirectangular) → skipped, logged.
- No landmarks and no mesh → exception with a clear message.
- Missing background/mask for a view when required → that view is skipped (2.0 behavior), logged by Blender.
- Blender non-zero exit → chunk fails; Blender's stdout/stderr is in the chunk log.

## Testing

1. **`tests/test_cameraMath.py`** (plain pytest, no Blender, no pyalicevision): build a synthetic pinhole camera (non-centered principal point, non-square pixels, rotated pose), project 3D points two ways — (a) AliceVision pinhole model `x = fx·(R(X−C))ₓ/z + w/2 + ox`, (b) Blender's model reconstructed from the `cameraMath` output (inverse `matrix_world`, perspective with lens/sensor/shift) — and assert equal pixels to 1e-6.
2. **Smoke render**: a rez command, run manually, that renders a small known dataset through the node in Meshroom; check that rendered landmarks overlay their features on the undistorted background images and that the mesh silhouette matches.

## Out of scope

Lens distortion, mesh formats other than `.obj`, textured/solid shading, depth/normal/label outputs.
