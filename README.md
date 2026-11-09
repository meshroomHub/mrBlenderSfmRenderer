# mrBlenderSfmRenderer
Blender plugin for rendering 3D models from multiple camera viewpoints using AliceVision SfM data.

## ScenePreview node

Renders a mesh (`.obj`), or the SfMData landmarks when no mesh is given, from every view of an SfMData.
Each render matches the view's pinhole intrinsics and pose (lens distortion is ignored: renders align with
undistorted images). The SfMData views are expected to be undistorted; each render is composited
over its view image, and masks can be applied.

How it works: the node's `preprocess` exports `cameras.json` (and `landmarks.ply`) with pyalicevision, then each
chunk runs `blender/render.py` in Blender. pyalicevision cannot be imported inside Blender, hence the split.

## Tests

```bash
rez env meshroom aliceVision pytest -- python -m pytest tests -v
# Blender tests are skipped unless BLENDER_CMD is set:
BLENDER_CMD=/path/to/blender rez env meshroom aliceVision pytest -- python -m pytest tests -v
```

`BLENDER_CMD` must launch Blender with its own environment (not the test environment's libraries).
With rez, use a wrapper script, since `rez env pkg -- cmd` drops arguments after a second `--`:

```bash
#!/bin/bash
exec rez env blender-4 -c "blender $(printf '%q ' "$@")"
```
