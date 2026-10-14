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

- **Mesh**: `bpy.ops.wm.obj_import(filepath, forward_axis='Y', up_axis='Z')` (file coordinates unchanged), then the y/z flip below. Wireframe or line-art shading as in 2.0.
- **Landmarks**: `bpy.ops.wm.ply_import(..., forward_axis='Y', up_axis='Z')`, then the 2.0 geometry-nodes particle setup.
- **World convention**: the scene is in sfmData world coordinates. Landmarks are exported from the sfmData unchanged. AliceVision meshes are saved with y and z negated (`Mesh::save`, `src/aliceVision/mesh/Mesh.cpp`), so the imported mesh object gets `matrix_world = diag(1, −1, −1, 1)`.
- Range: slice `views[rangeStart : rangeStart + rangeSize]`, clamped to the list length (as 2.0); skip entries with `valid: false`.
- Per view: apply `cameraMath` output, set background/mask image (same viewId/stem glob lookup as 2.0; skip the view if a required image is missing), render to `<output>/<imageStem>_preview.jpg`, free images. No rescale (see Camera math).

## Camera math (`cameraMath.py`)

AliceVision camera frame: x right, y down, z forward. Blender camera frame: x right, y up, looking along −z.

- `matrix_world = [[Rᵀ, C], [0, 1]] · diag(1, −1, −1, 1)`
- `sensor_fit = 'HORIZONTAL'`, `sensor_width = width` (pixel units), `lens = fx`
- `resolution = (width, height)`, `resolution_percentage = 100`
- Non-square pixels: `r = fy / fx`; `pixel_aspect = (r, 1)` if `r ≥ 1`, else `(1, 1/r)` (Blender clamps aspect to ≥ 1). The rendered image stays `width × height` with the right fx and fy, so there is **no post-render rescale** (2.0's rescale broke alignment with the undistorted images).
- `shift_x = −ox / width`, `shift_y = oy · fx / (fy · width)`

Verified on 2026-10-14 in Blender 4.5.3 with `bpy_extras.object_utils.world_to_camera_view`: landscape and portrait, `fy/fx` above and below 1, off-center principal point, rotated pose. Maximum error was 1.7e-4 px (float32).

## Error handling

- sfmData fails to load → exception in `preprocess`; the node fails before any Blender chunk runs.
- View without pose or intrinsic → skipped, logged.
- Non-pinhole intrinsic (equidistant, equirectangular) → skipped, logged.
- No landmarks and no mesh → exception in `preprocess` with a clear message.
- Missing background/mask for a view when required → that view is skipped (2.0 behavior), logged by Blender.
- Blender non-zero exit → chunk fails; Blender's stdout/stderr is in the chunk log (standard `CommandLineNode` behavior).

## Testing

Tests run in `rez env aliceVision python-3.11 pytest`; Blender is reached through the `BLENDER_CMD` environment variable (e.g. `rez env blender-4 -- blender`). Blender tests are skipped when it is unset.

1. **`tests/test_cameraMath.py`** (pure Python): checks the parameter formulas, camera center, and optical axis.
2. **`tests/test_sfmExport.py`** (pyalicevision): builds a synthetic `.sfm`, runs the export, and checks `cameras.json` (values, `valid: false` entries, sort order) and `landmarks.ply`.
3. **`tests/test_blenderProjection.py`** (Blender): applies `render.setupCamera` in Blender, projects points with `world_to_camera_view`, and compares them with the AliceVision pinhole projection (tolerance 1e-2 px).
4. **`tests/test_blenderRender.py`** (Blender): runs `render.py` end to end on a synthetic `cameras.json` + PLY, and checks one output image per valid view at `width × height`.
5. **Smoke run** in Meshroom on a real dataset, checked by eye: landmarks overlay their features on the undistorted backgrounds, and the mesh silhouette matches.

## Out of scope

Lens distortion, mesh formats other than `.obj`, textured/solid shading, depth/normal/label outputs.
