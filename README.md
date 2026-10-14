# mrBlenderSfmRenderer
Blender plugin for rendering 3D models from multiple camera viewpoints using AliceVision SfM data.

## ScenePreview node

Renders a mesh (`.obj`), or the SfMData landmarks when no mesh is given, from every view of an SfMData.
Each render matches the view's pinhole intrinsics and pose (lens distortion is ignored: renders align with
undistorted images). Undistorted images can be used as background and masks can be applied.

How it works: the node's `preprocess` exports `cameras.json` (and `landmarks.ply`) with pyalicevision, then each
chunk runs `scripts/render.py` in Blender. pyalicevision cannot be imported inside Blender, hence the split.

## Tests

```bash
rez env meshroom aliceVision pytest -- python -m pytest tests -v
# Blender tests are skipped unless BLENDER_CMD is set:
BLENDER_CMD="rez env blender-4 -- blender" rez env meshroom aliceVision pytest -- python -m pytest tests -v
```
