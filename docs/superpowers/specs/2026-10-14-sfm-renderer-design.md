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

`ScenePreview` stays a `desc.CommandLineNode`, like 2.0. The AliceVision work moves into the node's `preprocess`, which Meshroom runs once, as its own chunk (`preprocess.log`), before the parallel chunks:

```
preprocess(node)  — once, Meshroom compute process (pyalicevision)
  load sfmData → {nodeCacheFolder}/cameras.json  (+ landmarks.ply when no mesh)

chunk i  — parallel, plugin rez env (blender-4), command line
  blender -b --python render.py -- --views cameras.json [--landmarks landmarks.ply]
                                    {allParams} --rangeStart s --rangeSize n
```

For a `CommandLineNode`, `NodeChunk.process` calls `nodeDesc.preprocess(node)` in the current process (`meshroom/core/node.py`, `NodeChunk.process`). The node module already imports `pyalicevision` at module level, so pyalicevision is available there. Blender only runs in the chunk command line. The sfmData is loaded once, files are written once (no races between chunks), and there is no custom subprocess code.

### Files

| File | Role |
|---|---|
| `meshroom/blenderSfmRenderer/ScenePreview.py` | Node description, `commandLine`, and `preprocess` (pyalicevision export). |
| `meshroom/blenderSfmRenderer/scripts/render.py` | Blender script. Imports `bpy` and `cameraMath` only. Replaces `preview.py`. |
| `meshroom/blenderSfmRenderer/scripts/cameraMath.py` | Pure Python (no `bpy`, no pyalicevision): converts a pinhole camera to Blender camera parameters and `matrix_world` as nested lists. Unit-testable anywhere. |
| `meshroom/blenderSfmRenderer/__init__.py`, `scripts/__init__.py` | Restored from HEAD (currently deleted in the working tree). |
| `tests/test_cameraMath.py` | Unit tests (pytest). |

`preview.py` is deleted; its shading and compositing functions move to `render.py` mostly unchanged.

## Node interface

```python
commandLine = ("{blenderCmdValue} -b --python {nodeSourceCodeFolder}/scripts/render.py -- "
               "--views {nodeCacheFolder}/cameras.json --landmarks {nodeCacheFolder}/landmarks.ply {allParams}")
commandLineRange = "--rangeStart {rangeStart} --rangeSize {rangeBlockSize}"
size = avpar.DynamicViewsSize("cameras")
parallelization = desc.Parallelization(blockSize=40)
```

`render.py` uses `--landmarks` only when no `.obj` mesh is given.

Inputs (names kept from 2.0 where meaning is unchanged):

| Name | Type | Notes |
|---|---|---|
| `blenderCmd` | File | Default `blender`, `invalidate=False`, `commandLineGroup=""` (not in `allParams`). |
| `cameras` | File | sfmData, any pyalicevision-readable format. `commandLineGroup=""`: Blender reads `cameras.json`, not the sfmData. |
| `model` | File | Optional `.obj` mesh. Empty → render sfmData landmarks. A `.abc` value (from 2.0 graphs) logs a warning and falls back to landmarks. |
| `useBackground`, `undistortedImages` | Bool, File | Unchanged. |
| `useMasks`, `masks` | Bool, File | Unchanged (empty `masks` → background alpha, as in 2.0). |
| `pointCloudParams` (`particleSize`, `particleColor`) | Group | Enabled when `model` does not end with `.obj`. |
| `meshParams` (`shading`, `edgeColor`) | Group | Enabled when `model` ends with `.obj`. |

The `script` input is removed (the command line uses `{nodeSourceCodeFolder}`).

Outputs: unchanged from 2.0 — `output` = `{nodeCacheFolder}`, `frames` = `{nodeCacheFolder}/<FILESTEM>_preview.jpg` with `semantic="image"`.

## Stage 1 — `preprocess` (pyalicevision, once)

1. `sfmDataIO.load` the `cameras` file: views, intrinsics, extrinsics, plus structure only when landmarks are needed.
2. Iterate views sorted by `viewId`. This is the order the chunk ranges index into.
3. For each view, write an entry. Views without a pose or intrinsic, or with a non-pinhole intrinsic, get `"valid": false` and are logged. They are kept in the list so chunk ranges stay aligned with `DynamicViewsSize`.
4. A valid entry holds, from the `IntrinsicScaleOffset`: width, height, scale `(fx, fy)` in pixels, and offset `(ox, oy)` = principal point minus image center, in pixels. From the pose it holds rotation `R` (world→camera) and center `C`.
5. Write `{nodeCacheFolder}/cameras.json`:
   ```json
   { "views": [ { "viewId": "123", "valid": true, "imageStem": "IMG_0001",
                  "width": 4000, "height": 3000, "fx": 3200.0, "fy": 3200.0,
                  "ox": 12.3, "oy": -4.5, "R": [[...],[...],[...]], "C": [x, y, z] } ] }
   ```
6. If `model` does not end with `.obj`, write `{nodeCacheFolder}/landmarks.ply` (binary little-endian, `x y z` float + `red green blue` uchar). Raise if the sfmData has no landmarks.

## Stage 2 — `render.py` (bpy, per chunk)

Same scene setup as 2.0 (Cycles, 1 sample, no denoise, film transparent, compositing graph for background/mask).

- **Mesh**: `bpy.ops.wm.obj_import(filepath, forward_axis='Y', up_axis='Z')` — keeps AliceVision world coordinates unchanged. Wireframe or line-art shading as in 2.0.
- **Landmarks**: `bpy.ops.wm.ply_import(..., forward_axis='Y', up_axis='Z')`, then the 2.0 geometry-nodes particle setup.
- **World convention**: everything stays in AliceVision world coordinates; only the camera's local axes are converted.
- Range: slice `views[rangeStart : rangeStart + rangeSize]`, clamped to the list length (as 2.0); skip entries with `valid: false`.
- Per view: apply `cameraMath` output, set background/mask image (same viewId/stem glob lookup as 2.0; skip the view if a required image is missing), render to `<output>/<imageStem>_preview.jpg`, rescale if pixel ratio ≠ 1 (as 2.0), free images.

## Camera math (`cameraMath.py`)

AliceVision camera frame: x right, y down, z forward. Blender camera frame: x right, y up, looking along −z.

- `matrix_world = [[Rᵀ, C], [0, 1]] · diag(1, −1, −1, 1)`
- `sensor_fit = 'HORIZONTAL'`, `sensor_width = width` (pixel units), `lens = fx`
- `resolution = (width, height)`; pixel ratio `fy / fx` handled with `pixel_aspect_x` + post-rescale as in 2.0
- `shift_x = −ox / max(width, height)`, `shift_y = oy / max(width, height)` (Blender normalizes shift by the larger dimension; signs as in 2.0)

Correctness is defined by the projection test below, not by the formulas: if a sign is wrong the test fails and the formula is fixed.

## Error handling

- sfmData fails to load → exception in `preprocess`; the node fails before any Blender chunk runs.
- View without pose or intrinsic → skipped, logged.
- Non-pinhole intrinsic (equidistant, equirectangular) → skipped, logged.
- No landmarks and no mesh → exception in `preprocess` with a clear message.
- Missing background/mask for a view when required → that view is skipped (2.0 behavior), logged by Blender.
- Blender non-zero exit → chunk fails; Blender's stdout/stderr is in the chunk log (standard `CommandLineNode` behavior).

## Testing

1. **`tests/test_cameraMath.py`** (plain pytest, no Blender, no pyalicevision): build a synthetic pinhole camera (non-centered principal point, non-square pixels, rotated pose), project 3D points two ways — (a) AliceVision pinhole model `x = fx·(R(X−C))ₓ/z + w/2 + ox`, (b) Blender's model reconstructed from the `cameraMath` output (inverse `matrix_world`, perspective with lens/sensor/shift) — and assert equal pixels to 1e-6.
2. **Smoke render**: a rez command, run manually, that renders a small known dataset through the node in Meshroom; check that rendered landmarks overlay their features on the undistorted background images and that the mesh silhouette matches.

## Out of scope

Lens distortion, mesh formats other than `.obj`, textured/solid shading, depth/normal/label outputs.
