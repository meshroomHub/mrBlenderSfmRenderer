# ScenePreview 3.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the `ScenePreview` Meshroom node so that it renders a mesh, or the sfmData landmarks, with Blender from every view of an sfmData, matching each view's pinhole intrinsics and extrinsics exactly.

**Architecture:** `ScenePreview` stays a `desc.CommandLineNode`. Its `preprocess(node)` runs once in the Meshroom process and uses pyalicevision to export `cameras.json` (and `landmarks.ply` when there is no mesh) into the node cache folder. Each parallel chunk then runs `blender -b … --python scripts/render.py`. That script imports only `bpy` and the pure-Python `cameraMath`, and renders its range of views. pyalicevision cannot be loaded inside Blender, which is why it's split this way.

**Tech Stack:** Python 3.11, Meshroom `desc` API, pyalicevision (aliceVision 3.3.7), Blender 4.5 `bpy` (rez `blender-4`), numpy, pytest 8.

**Spec:** `docs/superpowers/specs/2026-10-14-sfm-renderer-design.md`

## Global Constraints

- pyalicevision must never be imported from `scripts/render.py` or `scripts/cameraMath.py` (it cannot load inside Blender).
- `scripts/cameraMath.py` imports neither `bpy` nor pyalicevision nor numpy: pure Python.
- Node class name stays `ScenePreview`; module `__version__ = "3.0"`.
- Output names unchanged from 2.0: `output` = `{nodeCacheFolder}`, `frames` = `{nodeCacheFolder}/<FILESTEM>_preview.jpg`, `semantic="image"`.
- `cameras.json` lists **all** views sorted by `viewId` (invalid ones with `"valid": false`) so chunk ranges match `avpar.DynamicViewsSize("cameras")`.
- Camera formulas (verified in Blender 4.5.3): `matrix_world = [[Rᵀ, C], [0, 1]] · diag(1, −1, −1, 1)`; `sensor_fit = 'HORIZONTAL'`; `lens / sensor_width = fx / width`; `r = fy/fx`, `pixel_aspect = (r, 1)` if `r ≥ 1` else `(1, 1/r)`; `shift_x = −ox / width`; `shift_y = oy · fx / (fy · width)`; no post-render rescale.
- Python test environment: `rez env meshroom aliceVision pytest -- python -m pytest …` (run from the repo root).
- Blender is reached through the env var `BLENDER_CMD`, e.g. `BLENDER_CMD="rez env blender-4 -- blender"`; Blender tests skip when it is unset.
- Code style: match the existing repo (camelCase functions and variables, `'''docstrings'''`, 4-space indent).

## Review Focus

1. **Empty optional file values from Meshroom** (`--model ""`, `--masks ""`, `--undistortedImages ""`). The script must treat them as "not given", not crash or look for a file named `""`. Pinned in Task 3 (`test_renderEmptyOptionalArgs`).
2. **A chunk range past the end of the view list** (last chunk, or views skipped). The render must clamp the range and exit 0. Pinned in Task 3 (`test_renderRangeClamped`).
3. **sfmData with landmarks requested but none present.** `preprocess` must raise a clear error before any Blender chunk runs. Pinned in Task 2 (`test_exportNoLandmarksRaises`).
4. **Non-square pixels and off-center principal point.** Renders must still align. Pinned in Task 3 (`test_blenderProjectionMatchesPinhole` parametrized with `fy != fx`).
5. **A `.abc` `model` value carried over from 2.0 graphs.** It must fall back to landmarks with a warning, not try to import the `.abc` as a mesh. Pinned in Task 4 (`test_preprocessAbcModelFallsBackToLandmarks`).

---

## File Structure

| File | Responsibility |
|---|---|
| `meshroom/blenderSfmRenderer/__init__.py` | Restore from HEAD (empty package marker). |
| `meshroom/blenderSfmRenderer/ScenePreview.py` | Node description, command line, `preprocess`. |
| `meshroom/blenderSfmRenderer/sfmExport.py` | pyalicevision → `cameras.json` / `landmarks.ply`. No Meshroom import. |
| `meshroom/blenderSfmRenderer/scripts/__init__.py` | Restore from HEAD. |
| `meshroom/blenderSfmRenderer/scripts/cameraMath.py` | Pinhole → Blender camera parameters (pure Python). |
| `meshroom/blenderSfmRenderer/scripts/render.py` | Blender script: scene, shading, compositing, render loop. |
| `meshroom/blenderSfmRenderer/scripts/preview.py` | Deleted (replaced by `render.py`). |
| `tests/conftest.py` | `sys.path` setup, `blenderCmd` fixture, `runBlender` and `jpegSize` helpers. |
| `tests/blenderProjectionCheck.py` | Script run inside Blender by the projection test. |
| `tests/test_cameraMath.py`, `tests/test_sfmExport.py`, `tests/test_blenderProjection.py`, `tests/test_blenderRender.py`, `tests/test_scenePreviewNode.py` | Tests. |
| `README.md` | Usage and test instructions. |

---

### Task 1: Test harness and `cameraMath`

**Files:**
- Restore: `meshroom/blenderSfmRenderer/__init__.py`, `meshroom/blenderSfmRenderer/scripts/__init__.py`
- Create: `tests/conftest.py`, `meshroom/blenderSfmRenderer/scripts/cameraMath.py`
- Test: `tests/test_cameraMath.py`

**Interfaces:**
- Produces: `cameraMath.blenderCamera(width, height, fx, fy, ox, oy, R, C) -> dict` with keys `resolution_x` (int), `resolution_y` (int), `pixel_aspect_x` (float), `pixel_aspect_y` (float), `sensor_fit` (`"HORIZONTAL"`), `sensor_width` (float), `lens` (float), `shift_x` (float), `shift_y` (float), `matrix_world` (4×4 nested lists). `R` is the 3×3 world→camera rotation as nested lists, `C` the camera center as a 3-list.
- Produces (conftest): fixture `blenderCmd` → `list[str]` (skips test when `BLENDER_CMD` is unset); `runBlender(blenderCmd, script, args) -> subprocess.CompletedProcess` (raises `AssertionError` with Blender output on non-zero exit); `jpegSize(path) -> (width, height)`; constants `REPO`, `NODE_DIR`, `SCRIPTS_DIR` (`pathlib.Path`).

- [ ] **Step 1: Restore the deleted package markers**

```bash
git restore meshroom/blenderSfmRenderer/__init__.py meshroom/blenderSfmRenderer/scripts/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
'''Shared test helpers.

Run from the repo root:
    rez env meshroom aliceVision pytest -- python -m pytest tests -v
Blender tests also need:
    export BLENDER_CMD="rez env blender-4 -- blender"
'''
import os
import pathlib
import shlex
import struct
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
NODE_DIR = REPO / "meshroom" / "blenderSfmRenderer"
SCRIPTS_DIR = NODE_DIR / "scripts"

# Import helper modules directly (the package name would clash with the real meshroom package).
for path in (str(NODE_DIR), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def blenderCmd():
    '''Command used to launch Blender, from the BLENDER_CMD environment variable.'''
    cmd = os.environ.get("BLENDER_CMD", "")
    if not cmd:
        pytest.skip("BLENDER_CMD is not set")
    return shlex.split(cmd)


def runBlender(blenderCmd, script, args):
    '''Run a Python script in background Blender and fail with its output on error.'''
    cmd = blenderCmd + ["-b", "--factory-startup", "--python-exit-code", "1",
                        "--python", str(script), "--"] + [str(a) for a in args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Blender failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return proc


def jpegSize(path):
    '''Return (width, height) read from the SOF marker of a JPEG file.'''
    with open(path, "rb") as f:
        data = f.read()
    assert data[:2] == b"\xff\xd8", f"{path} is not a JPEG"
    i = 2
    while i < len(data):
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    raise ValueError(f"No SOF marker in {path}")
```

- [ ] **Step 3: Write the failing tests `tests/test_cameraMath.py`**

```python
import math

import pytest

import cameraMath

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def rotationXYZ(ax, ay, az):
    '''World-to-camera rotation Rz(az) @ Ry(ay) @ Rx(ax) as nested lists.'''
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    mul = lambda a, b: [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mul(rz, mul(ry, rx))


def test_identityPoseFlipsYAndZ():
    p = cameraMath.blenderCamera(1000, 800, 900.0, 900.0, 0.0, 0.0, IDENTITY, [1.0, 2.0, 3.0])
    assert p["matrix_world"] == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 2.0],
        [0.0, 0.0, -1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_cameraCenterAndOpticalAxis():
    R = rotationXYZ(0.3, -0.7, 1.1)
    C = [0.5, -1.0, 2.0]
    m = cameraMath.blenderCamera(1000, 800, 900.0, 900.0, 0.0, 0.0, R, C)["matrix_world"]
    # Translation column is the camera center.
    assert [m[i][3] for i in range(3)] == pytest.approx(C)
    # Blender looks along local -Z: world direction = -(third column) = AliceVision optical axis = R's third row.
    assert [-m[i][2] for i in range(3)] == pytest.approx(R[2])
    # Blender local +Y (up) is AliceVision -y (image down): -(R's second row).
    assert [m[i][1] for i in range(3)] == pytest.approx([-v for v in R[1]])


def test_intrinsicsSquarePixels():
    p = cameraMath.blenderCamera(1000, 800, 900.0, 900.0, 20.0, -10.0, IDENTITY, [0, 0, 0])
    assert (p["resolution_x"], p["resolution_y"]) == (1000, 800)
    assert p["sensor_fit"] == "HORIZONTAL"
    assert p["lens"] / p["sensor_width"] == pytest.approx(900.0 / 1000)
    assert (p["pixel_aspect_x"], p["pixel_aspect_y"]) == (1.0, 1.0)
    assert p["shift_x"] == pytest.approx(-20.0 / 1000)
    assert p["shift_y"] == pytest.approx(-10.0 / 1000)


@pytest.mark.parametrize("fx, fy, aspect", [
    (900.0, 700.0, (1.0, 900.0 / 700.0)),   # fy < fx: stretch y (Blender aspect must be >= 1)
    (700.0, 900.0, (900.0 / 700.0, 1.0)),   # fy > fx: stretch x
])
def test_intrinsicsNonSquarePixels(fx, fy, aspect):
    p = cameraMath.blenderCamera(1000, 800, fx, fy, 20.0, -10.0, IDENTITY, [0, 0, 0])
    assert (p["pixel_aspect_x"], p["pixel_aspect_y"]) == pytest.approx(aspect)
    assert p["lens"] / p["sensor_width"] == pytest.approx(fx / 1000)
    assert p["shift_x"] == pytest.approx(-20.0 / 1000)
    assert p["shift_y"] == pytest.approx(-10.0 * fx / (fy * 1000))


def test_portraitUsesWidthForShift():
    p = cameraMath.blenderCamera(800, 1000, 900.0, 900.0, -15.0, 30.0, IDENTITY, [0, 0, 0])
    assert p["shift_x"] == pytest.approx(15.0 / 800)
    assert p["shift_y"] == pytest.approx(30.0 / 800)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `rez env meshroom aliceVision pytest -- python -m pytest tests/test_cameraMath.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'cameraMath'`.

- [ ] **Step 5: Write `meshroom/blenderSfmRenderer/scripts/cameraMath.py`**

```python
'''Convert an AliceVision pinhole camera to Blender camera settings.

Pure Python (no bpy, no pyalicevision) so it can be used inside Blender and in unit tests.
AliceVision camera frame: x right, y down, z forward.
Blender camera frame: x right, y up, looking along -z.
Formulas verified against bpy_extras.object_utils.world_to_camera_view (Blender 4.5).
'''

SENSOR_WIDTH = 36.0  # arbitrary; only lens / sensor_width matters


def blenderCamera(width, height, fx, fy, ox, oy, R, C):
    '''Return Blender render/camera settings matching a pinhole camera.

    width, height: image size in pixels.
    fx, fy: focal lengths in pixels.
    ox, oy: principal point minus image center, in pixels.
    R: 3x3 world-to-camera rotation (nested lists). C: camera center (3 values).
    '''
    ratio = fy / fx
    # Blender pixel aspect values must be >= 1
    if ratio >= 1.0:
        aspectX, aspectY = ratio, 1.0
    else:
        aspectX, aspectY = 1.0, 1.0 / ratio

    # matrix_world = [R^T | C] . diag(1, -1, -1, 1)
    matrixWorld = [
        [float(R[0][0]), -float(R[1][0]), -float(R[2][0]), float(C[0])],
        [float(R[0][1]), -float(R[1][1]), -float(R[2][1]), float(C[1])],
        [float(R[0][2]), -float(R[1][2]), -float(R[2][2]), float(C[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]

    return {
        "resolution_x": int(width),
        "resolution_y": int(height),
        "pixel_aspect_x": aspectX,
        "pixel_aspect_y": aspectY,
        "sensor_fit": "HORIZONTAL",
        "sensor_width": SENSOR_WIDTH,
        "lens": fx * SENSOR_WIDTH / width,
        # Blender shifts are normalized by the fitted (horizontal) size
        "shift_x": -ox / width,
        "shift_y": oy * fx / (fy * width),
        "matrix_world": matrixWorld,
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `rez env meshroom aliceVision pytest -- python -m pytest tests/test_cameraMath.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add meshroom/blenderSfmRenderer/__init__.py meshroom/blenderSfmRenderer/scripts/__init__.py \
        meshroom/blenderSfmRenderer/scripts/cameraMath.py tests/conftest.py tests/test_cameraMath.py
git commit -m "Add pinhole-to-Blender camera conversion with tests"
```

---

### Task 2: `sfmExport` (pyalicevision → `cameras.json`, `landmarks.ply`)

**Files:**
- Create: `meshroom/blenderSfmRenderer/sfmExport.py`
- Test: `tests/test_sfmExport.py`

**Interfaces:**
- Consumes: conftest `sys.path` setup (Task 1).
- Produces: `sfmExport.exportScene(sfmPath: str, camerasPath: str, landmarksPath: str | None = None, logger=None) -> dict`. It returns `{"views": int, "valid": int}`, writes `camerasPath` (JSON `{"views": [entry, ...]}`), and writes `landmarksPath` when it is not `None`. It raises `RuntimeError` if the sfmData can't be loaded, or if `landmarksPath` is given and the sfmData has no landmarks. `logger` is anything with `.info` and `.warning` (defaults to the `logging` module).
- Entry keys: always `viewId` (str), `imageStem` (str), `valid` (bool). Valid entries also have `width`, `height` (int), `fx`, `fy`, `ox`, `oy` (float), `R` (3×3 lists, world→camera), `C` (3-list).
- Produces: `sfmExport.isPinholeFamily(typeStr: str) -> bool`.
- PLY format: `binary_little_endian 1.0`, one `vertex` element with `float x, y, z` and `uchar red, green, blue`.

- [ ] **Step 1: Write the failing tests `tests/test_sfmExport.py`**

```python
import json

import numpy as np
import pytest

import sfmExport

R = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
C = [1.0, 2.0, 3.0]
LANDMARKS = [([1.5, 2.5, 3.5], [10, 20, 30]), ([-1.0, 0.0, 4.0], [255, 0, 128])]


def makeSfm(path, withLandmarks=True):
    '''Write a small sfmData: view 10 posed (pinhole), view 5 without pose.

    Landmarks cannot be created from Python (EImageDescriberType is not bound),
    so they are added to the saved JSON directly.
    '''
    from pyalicevision import sfmData, sfmDataIO, camera, geometry
    sfm = sfmData.SfMData()
    sfm.getIntrinsics()[1] = camera.Pinhole(1000, 800, 900, 700, 5, -3)
    posed = sfmData.View("/data/IMG_0010.jpg", 10, 1, 20, 1000, 800)
    sfm.getViews()[10] = posed
    sfm.setPose(posed, sfmData.CameraPose(geometry.Pose3(np.array(R), np.array(C))))
    sfm.getViews()[5] = sfmData.View("/data/IMG_0005.jpg", 5, 1, sfmData.UndefinedIndexT, 1000, 800)
    assert sfmDataIO.save(sfm, str(path), sfmDataIO.ALL)
    if withLandmarks:
        data = json.loads(path.read_text())
        data["structure"] = [
            {"landmarkId": str(i), "descType": "sift", "color": [str(c) for c in rgb],
             "X": [str(x) for x in X], "observations": []}
            for i, (X, rgb) in enumerate(LANDMARKS)
        ]
        path.write_text(json.dumps(data))
    return path


def readPly(path):
    '''Return (header lines, structured array) of a binary little-endian PLY.'''
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    return header, np.frombuffer(raw[end:], dtype=dtype)


def test_isPinholeFamily():
    for t in ["pinhole", "radial3", "brown", "fisheye4", "3deanamorphic4"]:
        assert sfmExport.isPinholeFamily(t)
    for t in ["equidistant_r3", "equirectangular"]:
        assert not sfmExport.isPinholeFamily(t)


def test_exportCameras(tmp_path):
    sfmPath = makeSfm(tmp_path / "in.sfm", withLandmarks=False)
    summary = sfmExport.exportScene(str(sfmPath), str(tmp_path / "cameras.json"))
    assert summary == {"views": 2, "valid": 1}

    views = json.loads((tmp_path / "cameras.json").read_text())["views"]
    # Sorted by viewId, invalid views kept so chunk ranges stay aligned
    assert [v["viewId"] for v in views] == ["5", "10"]
    assert views[0] == {"viewId": "5", "imageStem": "IMG_0005", "valid": False}

    v = views[1]
    assert v["valid"] is True
    assert v["imageStem"] == "IMG_0010"
    assert (v["width"], v["height"]) == (1000, 800)
    assert [v["fx"], v["fy"], v["ox"], v["oy"]] == pytest.approx([900.0, 700.0, 5.0, -3.0])
    assert np.allclose(v["R"], R)
    assert v["C"] == pytest.approx(C)
    assert not (tmp_path / "landmarks.ply").exists()


def test_exportLandmarks(tmp_path):
    sfmPath = makeSfm(tmp_path / "in.sfm")
    sfmExport.exportScene(str(sfmPath), str(tmp_path / "cameras.json"), str(tmp_path / "landmarks.ply"))
    header, points = readPly(tmp_path / "landmarks.ply")
    assert header[1] == "format binary_little_endian 1.0"
    assert "element vertex 2" in header
    got = sorted((tuple(np.round([p["x"], p["y"], p["z"]], 4)), (p["red"], p["green"], p["blue"])) for p in points)
    expected = sorted((tuple(X), tuple(rgb)) for X, rgb in LANDMARKS)
    assert got == expected


def test_exportNoLandmarksRaises(tmp_path):
    sfmPath = makeSfm(tmp_path / "in.sfm", withLandmarks=False)
    with pytest.raises(RuntimeError, match="no landmarks"):
        sfmExport.exportScene(str(sfmPath), str(tmp_path / "cameras.json"), str(tmp_path / "landmarks.ply"))


def test_exportBadFileRaises(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot load"):
        sfmExport.exportScene(str(tmp_path / "missing.sfm"), str(tmp_path / "cameras.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `rez env meshroom aliceVision pytest -- python -m pytest tests/test_sfmExport.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sfmExport'`.

- [ ] **Step 3: Write `meshroom/blenderSfmRenderer/sfmExport.py`**

```python
'''Export an sfmData to plain files read by the Blender render script.

Runs in the Meshroom process: pyalicevision cannot be imported inside Blender.
'''
import json
import logging
import os

import numpy as np

PLY_DTYPE = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]


def isPinholeFamily(typeStr):
    '''Equidistant and equirectangular intrinsics cannot be rendered with a perspective camera.'''
    return not typeStr.startswith("equi")


def viewEntry(sfm, view):
    '''Return (entry, reason) for a view; reason is None when the view can be rendered.'''
    imagePath = view.getImage().getImagePath()
    entry = {
        "viewId": str(view.getViewId()),
        "imageStem": os.path.splitext(os.path.basename(imagePath))[0],
        "valid": False,
    }
    if not sfm.isPoseAndIntrinsicDefined(view):
        return entry, "no pose or intrinsic"

    intrinsic = sfm.getIntrinsics()[view.getIntrinsicId()]
    if not isPinholeFamily(intrinsic.getTypeStr()):
        return entry, f"unsupported intrinsic type '{intrinsic.getTypeStr()}'"
    # IntrinsicScaleOffset parameters start with scale (fx, fy) then offset (ox, oy)
    fx, fy, ox, oy = list(intrinsic.getParameters())[:4]

    # Keep the CameraPose alive: rotation() returns a view into its memory
    pose = sfm.getPose(view)
    transform = pose.getTransform()
    rotation = np.array(transform.rotation(), dtype=float)
    center = np.array(transform.center(), dtype=float).ravel()

    entry.update(
        valid=True,
        width=int(intrinsic.w()),
        height=int(intrinsic.h()),
        fx=float(fx), fy=float(fy), ox=float(ox), oy=float(oy),
        R=rotation.tolist(),
        C=center.tolist(),
    )
    return entry, None


def writeLandmarksPly(sfm, path):
    '''Write landmark positions and colors as a binary PLY point cloud.'''
    landmarks = sfm.getLandmarks()
    points = np.zeros(len(landmarks), dtype=PLY_DTYPE)
    for i, (_, landmark) in enumerate(landmarks.items()):
        X = np.asarray(landmark.X, dtype=float).ravel()
        rgb = np.asarray(landmark.rgb).ravel()
        points[i] = (X[0], X[1], X[2], rgb[0], rgb[1], rgb[2])
    if len(points) == 0:
        raise RuntimeError("The sfmData has no landmarks and no mesh was given: nothing to render.")

    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        points.tofile(f)


def exportScene(sfmPath, camerasPath, landmarksPath=None, logger=None):
    '''Write cameras.json (all views, sorted by viewId) and optionally landmarks.ply.'''
    from pyalicevision import sfmData, sfmDataIO

    logger = logger or logging
    parts = sfmDataIO.VIEWS | sfmDataIO.INTRINSICS | sfmDataIO.EXTRINSICS
    if landmarksPath:
        parts |= sfmDataIO.STRUCTURE

    sfm = sfmData.SfMData()
    if not os.path.isfile(sfmPath) or not sfmDataIO.load(sfm, sfmPath, parts):
        raise RuntimeError(f"Cannot load sfmData: {sfmPath}")

    entries = []
    for viewId, view in sorted(sfm.getViews().items(), key=lambda item: item[0]):
        entry, reason = viewEntry(sfm, view)
        if reason:
            logger.warning(f"View {viewId} will not be rendered: {reason}")
        entries.append(entry)

    with open(camerasPath, "w") as f:
        json.dump({"views": entries}, f, indent=1)

    if landmarksPath:
        writeLandmarksPly(sfm, landmarksPath)

    nbValid = sum(1 for e in entries if e["valid"])
    logger.info(f"Exported {nbValid}/{len(entries)} renderable views to {camerasPath}")
    return {"views": len(entries), "valid": nbValid}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `rez env meshroom aliceVision pytest -- python -m pytest tests/test_sfmExport.py -v`
Expected: 5 passed. If `sfmDataIO.load` rejects the view with `UndefinedIndexT` pose, check that `sfmData.UndefinedIndexT` exists (it is used by AliceVision's `SfMRigApplying` node). Do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add meshroom/blenderSfmRenderer/sfmExport.py tests/test_sfmExport.py
git commit -m "Add pyalicevision export of cameras and landmarks for Blender"
```

---

### Task 3: Blender render script `render.py`

**Files:**
- Create: `meshroom/blenderSfmRenderer/scripts/render.py`, `tests/blenderProjectionCheck.py`
- Delete: `meshroom/blenderSfmRenderer/scripts/preview.py` (already deleted in the working tree; stage the deletion)
- Test: `tests/test_blenderProjection.py`, `tests/test_blenderRender.py`

**Interfaces:**
- Consumes: `cameraMath.blenderCamera` (Task 1); the `cameras.json` entry format and PLY format (Task 2); conftest `blenderCmd`, `runBlender`, `jpegSize`, `SCRIPTS_DIR`, `REPO`.
- Produces: `render.setupCamera(scene, camObj, view: dict) -> None`. It applies one `cameras.json` entry to the scene render settings and camera. `tests/blenderProjectionCheck.py` imports it.
- Produces: the CLI used by the node (Task 4):
  - required: `--views FILE`, `--output DIR`;
  - optional: `--landmarks FILE`, `--model FILE`, `--useBackground True|False`, `--undistortedImages DIR`, `--useMasks True|False`, `--masks DIR`, `--particleSize F`, `--particleColor NAME`, `--edgeColor NAME`, `--shading wireframe|line_art`, `--rangeStart N`, `--rangeSize N`.
  - Empty strings for the file options mean "not given".
  - Writes `<output>/<imageStem>_preview.jpg` for each valid view in range.

- [ ] **Step 1: Write `tests/blenderProjectionCheck.py`** (runs inside Blender)

```python
'''Run inside Blender: apply render.setupCamera and project points with world_to_camera_view.

Usage: blender -b --python blenderProjectionCheck.py -- <scriptsDir> <input.json> <output.json>
input.json: {"views": [cameras.json entries], "points": [[x, y, z], ...]}
output.json: {"<viewId>": [[u, v], ...]} in pixels, origin top-left.
'''
import json
import sys

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, argv[0])
import render  # noqa: E402

with open(argv[1]) as f:
    data = json.load(f)

scene = bpy.context.scene
result = {}
for view in data["views"]:
    render.setupCamera(scene, scene.camera, view)
    bpy.context.view_layer.update()
    pixels = []
    for X in data["points"]:
        p = world_to_camera_view(scene, scene.camera, mathutils.Vector(X))
        pixels.append([p.x * view["width"], (1.0 - p.y) * view["height"]])
    result[view["viewId"]] = pixels

with open(argv[2], "w") as f:
    json.dump(result, f)
```

- [ ] **Step 2: Write the failing test `tests/test_blenderProjection.py`**

```python
import json

import numpy as np
import pytest

from conftest import REPO, SCRIPTS_DIR, runBlender


def rotation(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize("width, height, fx, fy, ox, oy", [
    (1000, 800, 900.0, 900.0, 20.0, -10.0),
    (800, 1000, 900.0, 900.0, 20.0, -10.0),    # portrait
    (1000, 800, 900.0, 700.0, 20.0, -10.0),    # fy < fx
    (800, 1000, 650.0, 900.0, -15.0, 30.0),    # fy > fx, portrait
])
def test_blenderProjectionMatchesPinhole(tmp_path, blenderCmd, width, height, fx, fy, ox, oy):
    R = rotation(0.3, -0.7, 1.1)
    C = np.array([0.5, -1.0, 2.0])
    camPoints = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 3.0], [0.0, 0.0, 1.0], [1.0, 0.8, 5.0]])
    worldPoints = (R.T @ camPoints.T).T + C
    view = {"viewId": "1", "imageStem": "img", "valid": True, "width": width, "height": height,
            "fx": fx, "fy": fy, "ox": ox, "oy": oy, "R": R.tolist(), "C": C.tolist()}
    (tmp_path / "in.json").write_text(json.dumps({"views": [view], "points": worldPoints.tolist()}))

    runBlender(blenderCmd, REPO / "tests" / "blenderProjectionCheck.py",
               [SCRIPTS_DIR, tmp_path / "in.json", tmp_path / "out.json"])

    got = np.array(json.loads((tmp_path / "out.json").read_text())["1"])
    expected = np.stack([fx * camPoints[:, 0] / camPoints[:, 2] + width / 2 + ox,
                         fy * camPoints[:, 1] / camPoints[:, 2] + height / 2 + oy], axis=1)
    assert np.abs(got - expected).max() < 1e-2
```

- [ ] **Step 3: Write the failing test `tests/test_blenderRender.py`**

```python
import json

import numpy as np

from conftest import SCRIPTS_DIR, jpegSize, runBlender

PLY_DTYPE = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def writeScene(folder):
    '''cameras.json with an invalid view and two valid views of different sizes, plus a PLY.'''
    views = [
        {"viewId": "1", "imageStem": "skipped", "valid": False},
        {"viewId": "2", "imageStem": "imgA", "valid": True, "width": 320, "height": 240,
         "fx": 300.0, "fy": 300.0, "ox": 5.0, "oy": -4.0, "R": IDENTITY, "C": [0.0, 0.0, -5.0]},
        {"viewId": "3", "imageStem": "imgB", "valid": True, "width": 200, "height": 300,
         "fx": 250.0, "fy": 260.0, "ox": 0.0, "oy": 0.0, "R": IDENTITY, "C": [0.2, 0.0, -6.0]},
    ]
    (folder / "cameras.json").write_text(json.dumps({"views": views}))
    points = np.zeros(3, dtype=PLY_DTYPE)
    points[:] = [(0, 0, 0, 255, 0, 0), (0.5, 0.2, 0.1, 0, 255, 0), (-0.4, -0.3, 0.2, 0, 0, 255)]
    with open(folder / "landmarks.ply", "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        points.tofile(f)


def baseArgs(folder, out):
    return ["--views", folder / "cameras.json", "--landmarks", folder / "landmarks.ply",
            "--output", out, "--useBackground", "False", "--useMasks", "False",
            "--particleSize", "0.01", "--particleColor", "Red"]


def test_renderLandmarksAllViews(tmp_path, blenderCmd):
    writeScene(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py", baseArgs(tmp_path, out) + ["--model", ""])
    assert sorted(p.name for p in out.iterdir()) == ["imgA_preview.jpg", "imgB_preview.jpg"]
    assert jpegSize(out / "imgA_preview.jpg") == (320, 240)
    assert jpegSize(out / "imgB_preview.jpg") == (200, 300)  # no post-render rescale


def test_renderRangeClamped(tmp_path, blenderCmd):
    writeScene(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py",
               baseArgs(tmp_path, out) + ["--rangeStart", "2", "--rangeSize", "40"])
    assert sorted(p.name for p in out.iterdir()) == ["imgB_preview.jpg"]


def test_renderEmptyOptionalArgs(tmp_path, blenderCmd):
    '''Meshroom passes empty quoted values for unset File attributes.'''
    writeScene(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py",
               baseArgs(tmp_path, out) + ["--model", "", "--masks", "", "--undistortedImages", ""])
    assert len(list(out.iterdir())) == 2


def test_renderMeshWireframe(tmp_path, blenderCmd):
    writeScene(tmp_path)
    (tmp_path / "tri.obj").write_text("v -1 -1 0\nv 1 -1 0\nv 0 1 0\nf 1 2 3\n")
    out = tmp_path / "out"
    out.mkdir()
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py",
               baseArgs(tmp_path, out) + ["--model", tmp_path / "tri.obj", "--shading", "wireframe",
                                          "--edgeColor", "Green"])
    assert jpegSize(out / "imgA_preview.jpg") == (320, 240)
```

- [ ] **Step 4: Run the Blender tests to verify they fail**

Run: `BLENDER_CMD="rez env blender-4 -- blender" rez env meshroom aliceVision pytest -- python -m pytest tests/test_blenderProjection.py tests/test_blenderRender.py -v`
Expected: all FAIL. Blender exits non-zero with `ModuleNotFoundError: No module named 'render'`, or `render.py` is not found.

- [ ] **Step 5: Write `meshroom/blenderSfmRenderer/scripts/render.py`**

The shading and compositing functions are ported from the deleted `preview.py` (`git show HEAD:meshroom/blenderSfmRenderer/scripts/preview.py`). Compared with it, this version:
- takes cameras from `cameraMath`;
- loads landmarks from a PLY;
- treats empty option values as "not given";
- does no post-render rescale;
- reads freestyle settings from `bpy.context.view_layer` (`bpy.context.window` is `None` in background mode);
- spells `use_adaptive_sampling` correctly.

```python
import argparse
import glob
import json
import os
import sys

import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cameraMath  # noqa: E402


def strToBool(value):
    return str(value).lower() == 'true'


def createParser():
    '''Create command line interface.'''
    usage_text = (
        "Run blender in background mode with this script:"
        "  blender --background --python " + __file__ + " -- [options]"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # File options may be given empty by Meshroom: empty means "not given"
    optionalFile = dict(nargs='?', const='', default='')

    parser.add_argument("--views", metavar='FILE', required=True,
                        help="cameras.json exported from the sfmData.")
    parser.add_argument("--landmarks", metavar='FILE', **optionalFile,
                        help="Landmarks PLY, rendered when no mesh is given.")
    parser.add_argument("--model", metavar='FILE', **optionalFile,
                        help="Mesh (.obj) to render. Anything else renders the landmarks.")
    parser.add_argument("--useBackground", type=strToBool, default=False,
                        help="Display the undistorted image as background.")
    parser.add_argument("--undistortedImages", metavar='DIR', **optionalFile,
                        help="Folder containing undistorted images to use as background.")
    parser.add_argument("--useMasks", type=strToBool, default=False,
                        help="Apply mask to the rendered geometry.")
    parser.add_argument("--masks", metavar='DIR', **optionalFile,
                        help="Folder containing masks. Empty with useMasks: use background image alpha.")
    parser.add_argument("--particleSize", type=float, default=0.01,
                        help="Scale of particles used to show the point cloud.")
    parser.add_argument("--particleColor", type=str, default='Red',
                        help="Color of particles used to show the point cloud.")
    parser.add_argument("--edgeColor", type=str, default='Red',
                        help="Color of the edges of the mesh.")
    parser.add_argument("--shading", type=str, default='wireframe',
                        help="Shading method for rendering the mesh (wireframe, line_art).")
    parser.add_argument("--rangeStart", type=int, default=-1,
                        help="Range start for processing views. Set to -1 to process all views.")
    parser.add_argument("--rangeSize", type=int, default=0,
                        help="Range size for processing views.")
    parser.add_argument("--output", metavar='DIR', required=True,
                        help="Folder where rendered images are written.")
    return parser


# Color palette (common for point cloud and mesh visualization)
PALETTE = {
    'Grey': (0.2, 0.2, 0.2, 1),
    'White': (1, 1, 1, 1),
    'Red': (0.5, 0, 0, 1),
    'Green': (0, 0.5, 0, 1),
    'Magenta': (1.0, 0, 0.75, 1),
}


def setupCamera(scene, camObj, view):
    '''Setup Blender render settings and camera to match a cameras.json view entry.'''
    params = cameraMath.blenderCamera(view['width'], view['height'], view['fx'], view['fy'],
                                      view['ox'], view['oy'], view['R'], view['C'])
    scene.render.resolution_x = params['resolution_x']
    scene.render.resolution_y = params['resolution_y']
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = params['pixel_aspect_x']
    scene.render.pixel_aspect_y = params['pixel_aspect_y']

    camData = camObj.data
    camData.clip_end = 10000.0
    camData.sensor_fit = params['sensor_fit']
    camData.sensor_width = params['sensor_width']
    camData.lens = params['lens']
    camData.shift_x = params['shift_x']
    camData.shift_y = params['shift_y']
    camObj.matrix_world = mathutils.Matrix(params['matrix_world'])


def initScene():
    '''Initialize Blender scene (factory startup: default camera, cube and light).'''
    for name in ('Cube', 'Light'):
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name])
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False


def initCompositing(useBackground, useMasks, useBackgroundAlpha=False):
    '''Initialize Blender compositing graph for adding background image and mask to render.'''
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    nodeAlphaOver = tree.nodes.new(type="CompositorNodeAlphaOver")
    nodeSetAlpha = tree.nodes.new(type="CompositorNodeSetAlpha")
    nodeBackground = tree.nodes.new(type="CompositorNodeImage")
    nodeMask = tree.nodes.new(type="CompositorNodeImage")
    nodeRender = tree.nodes['Render Layers']
    nodeComposite = tree.nodes['Composite']
    links = tree.links
    if useBackground and useMasks:
        links.new(nodeBackground.outputs['Image'], nodeAlphaOver.inputs[1])
        links.new(nodeRender.outputs['Image'], nodeSetAlpha.inputs['Image'])
        if useBackgroundAlpha:
            links.new(nodeBackground.outputs['Alpha'], nodeSetAlpha.inputs['Alpha'])
        else:
            links.new(nodeMask.outputs['Image'], nodeSetAlpha.inputs['Alpha'])
        links.new(nodeSetAlpha.outputs['Image'], nodeAlphaOver.inputs[2])
        links.new(nodeAlphaOver.outputs['Image'], nodeComposite.inputs['Image'])
    elif useBackground:
        links.new(nodeBackground.outputs['Image'], nodeAlphaOver.inputs[1])
        links.new(nodeRender.outputs['Image'], nodeAlphaOver.inputs[2])
        links.new(nodeAlphaOver.outputs['Image'], nodeComposite.inputs['Image'])
    elif useMasks:
        links.new(nodeRender.outputs['Image'], nodeSetAlpha.inputs['Image'])
        if useBackgroundAlpha:
            # No background to take alpha from: fall back to render alpha
            links.new(nodeRender.outputs['Alpha'], nodeSetAlpha.inputs['Alpha'])
        else:
            links.new(nodeMask.outputs['Image'], nodeSetAlpha.inputs['Alpha'])
        links.new(nodeSetAlpha.outputs['Image'], nodeComposite.inputs['Image'])
    return nodeBackground, nodeMask


def findImage(folder, view):
    '''Find the image of a view in a folder, by viewId then by image name.'''
    for key in (view['viewId'], view['imageStem']):
        matches = sorted(glob.glob(os.path.join(glob.escape(folder), '*' + glob.escape(key) + '*')))
        if matches:
            return matches[0]
    return None


def loadImageInNode(folder, view, node):
    '''Load the view image found in folder into a compositing image node. Return the image or None.'''
    path = findImage(folder, view)
    if not path:
        return None
    img = bpy.data.images.load(filepath=path)
    node.image = img
    return img


def importSelected(operator, filepath):
    '''Run an import operator keeping file coordinates unchanged and return the imported object.'''
    bpy.ops.object.select_all(action='DESELECT')
    operator(filepath=filepath, forward_axis='Y', up_axis='Z')
    return bpy.context.selected_objects[0]


def setupWireframeShading(mesh, color):
    '''Setup material for wireframe shading.'''
    material = bpy.data.materials.new('Wireframe')
    material.use_backface_culling = True
    material.use_nodes = True
    material.blend_method = 'BLEND'
    material.node_tree.links.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodeGeometry = nodes.new(type='ShaderNodeNewGeometry')
    nodeWireframe = nodes.new(type='ShaderNodeWireframe')
    nodeWireframe.use_pixel_size = True
    nodeWireframe.inputs['Size'].default_value = 2.0
    nodeEmission = nodes.new(type='ShaderNodeEmission')
    nodeEmission.inputs['Color'].default_value = color
    nodeHoldout = nodes.new(type='ShaderNodeHoldout')
    nodeMix1 = nodes.new(type='ShaderNodeMixShader')
    nodeTransparent = nodes.new(type='ShaderNodeBsdfTransparent')
    nodeMix2 = nodes.new(type='ShaderNodeMixShader')
    nodeOutput = nodes['Material Output']
    links.new(nodeWireframe.outputs['Fac'], nodeMix1.inputs['Fac'])
    links.new(nodeHoldout.outputs['Holdout'], nodeMix1.inputs[1])
    links.new(nodeEmission.outputs['Emission'], nodeMix1.inputs[2])
    links.new(nodeGeometry.outputs['Backfacing'], nodeMix2.inputs['Fac'])
    links.new(nodeMix1.outputs['Shader'], nodeMix2.inputs[1])
    links.new(nodeTransparent.outputs['BSDF'], nodeMix2.inputs[2])
    links.new(nodeMix1.outputs['Shader'], nodeOutput.inputs['Surface'])
    mesh.materials.clear()
    mesh.materials.append(material)


def setupLineArtShading(mesh, color):
    '''Setup line art shading using Freestyle.'''
    bpy.context.scene.render.use_freestyle = True
    bpy.data.linestyles["LineStyle"].color = (color[0], color[1], color[2])
    # See through one surface:
    # https://docs.blender.org/manual/id/3.3/render/freestyle/view_layer/line_set.html
    lineset = bpy.context.view_layer.freestyle_settings.linesets["LineSet"]
    lineset.visibility = "RANGE"
    lineset.qi_start = 0
    lineset.qi_end = 1
    material = bpy.data.materials.new('Holdout')
    material.use_nodes = True
    material.node_tree.links.clear()
    nodeHoldout = material.node_tree.nodes.new(type='ShaderNodeHoldout')
    nodeOutput = material.node_tree.nodes['Material Output']
    material.node_tree.links.new(nodeHoldout.outputs['Holdout'], nodeOutput.inputs['Surface'])
    mesh.materials.clear()
    mesh.materials.append(material)


def setupPointCloudShading(obj, color, size):
    '''Setup material and geometry nodes showing each point as a cube at unit distance from the camera.'''
    material = bpy.data.materials.new('PointCloud_Mat')
    material.use_nodes = True
    material.node_tree.links.clear()
    nodeEmission = material.node_tree.nodes.new(type='ShaderNodeEmission')
    nodeEmission.inputs['Color'].default_value = color
    nodeOutputFill = material.node_tree.nodes['Material Output']
    material.node_tree.links.new(nodeEmission.outputs['Emission'], nodeOutputFill.inputs['Surface'])

    geo = bpy.data.node_groups.new('Particles_Graph', type='GeometryNodeTree')
    mod = obj.modifiers.new('Particles_Modifier', type='NODES')
    mod.node_group = geo

    nodeActiveCamera = geo.nodes.new(type='GeometryNodeInputActiveCamera')
    nodeCameraObjectInfo = geo.nodes.new(type='GeometryNodeObjectInfo')
    nodeObjectPosition = geo.nodes.new(type='GeometryNodeInputPosition')
    nodeSubtract = geo.nodes.new(type='ShaderNodeVectorMath')
    nodeNormalize = geo.nodes.new(type='ShaderNodeVectorMath')
    nodeAdd = geo.nodes.new(type='ShaderNodeVectorMath')
    nodeSetPosition = geo.nodes.new(type='GeometryNodeSetPosition')
    nodeSubtract.operation = 'SUBTRACT'
    nodeNormalize.operation = 'NORMALIZE'
    nodeAdd.operation = 'ADD'
    nodeCameraObjectInfo.transform_space = 'RELATIVE'

    nodeInput = geo.nodes.new(type='NodeGroupInput')
    nodeOutput = geo.nodes.new(type='NodeGroupOutput')
    nodeIoP = geo.nodes.new(type='GeometryNodeInstanceOnPoints')
    nodeCube = geo.nodes.new(type='GeometryNodeMeshCube')
    nodeSize = geo.nodes.new(type='ShaderNodeValue')
    nodeSize.outputs['Value'].default_value = size
    nodeMat = geo.nodes.new(type='GeometryNodeSetMaterial')
    nodeMat.inputs[2].default_value = material

    geo.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    geo.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    geo.links.new(nodeActiveCamera.outputs['Active Camera'], nodeCameraObjectInfo.inputs['Object'])
    geo.links.new(nodeObjectPosition.outputs['Position'], nodeSubtract.inputs[0])
    geo.links.new(nodeCameraObjectInfo.outputs['Location'], nodeSubtract.inputs[1])
    geo.links.new(nodeSubtract.outputs['Vector'], nodeNormalize.inputs['Vector'])
    geo.links.new(nodeNormalize.outputs['Vector'], nodeAdd.inputs[0])
    geo.links.new(nodeCameraObjectInfo.outputs['Location'], nodeAdd.inputs[1])
    geo.links.new(nodeInput.outputs['Geometry'], nodeSetPosition.inputs['Geometry'])
    geo.links.new(nodeAdd.outputs['Vector'], nodeSetPosition.inputs['Position'])
    geo.links.new(nodeSetPosition.outputs['Geometry'], nodeIoP.inputs['Points'])
    geo.links.new(nodeCube.outputs['Mesh'], nodeIoP.inputs['Instance'])
    geo.links.new(nodeSize.outputs['Value'], nodeIoP.inputs['Scale'])
    geo.links.new(nodeIoP.outputs['Instances'], nodeMat.inputs['Geometry'])
    geo.links.new(nodeMat.outputs[0], nodeOutput.inputs['Geometry'])


def selectRange(views, rangeStart, rangeSize):
    '''Return the views of the chunk range, clamped to the list length.'''
    if rangeStart < 0:
        return views
    return views[rangeStart:rangeStart + max(rangeSize, 0)]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = createParser()
    if not argv:
        parser.print_help()
        return 1
    args = parser.parse_args(argv)

    if args.useBackground and not args.undistortedImages:
        print("Error: --undistortedImages is required with --useBackground True.")
        return 1

    useMesh = args.model.lower().endswith('.obj')
    if not useMesh and not args.landmarks:
        print("Error: no .obj mesh and no --landmarks given: nothing to render.")
        return 1

    # useMasks without a masks folder: use the background image alpha channel
    useBackgroundAlpha = args.useMasks and not args.masks

    print("Init scene")
    initScene()
    scene = bpy.context.scene
    nodeBackground, nodeMask = initCompositing(args.useBackground, args.useMasks, useBackgroundAlpha)

    with open(args.views) as f:
        views = json.load(f)['views']

    if useMesh:
        print("Load mesh " + args.model)
        obj = importSelected(bpy.ops.wm.obj_import, args.model)
        if args.shading == 'line_art':
            setupLineArtShading(obj.data, PALETTE[args.edgeColor])
        else:
            setupWireframeShading(obj.data, PALETTE[args.edgeColor])
    else:
        print("Load landmarks " + args.landmarks)
        obj = importSelected(bpy.ops.wm.ply_import, args.landmarks)
        setupPointCloudShading(obj, PALETTE[args.particleColor], args.particleSize)

    for view in selectRange(views, args.rangeStart, args.rangeSize):
        if not view['valid']:
            continue
        print("Rendering view " + view['viewId'])

        img = None
        if args.useBackground:
            img = loadImageInNode(args.undistortedImages, view, nodeBackground)
            if not img:
                print("No background image for view " + view['viewId'] + ", skipped")
                continue

        mask = None
        if args.useMasks and not useBackgroundAlpha:
            mask = loadImageInNode(args.masks, view, nodeMask)
            if not mask:
                print("No mask for view " + view['viewId'] + ", skipped")
                if img:
                    bpy.data.images.remove(img)
                continue

        setupCamera(scene, scene.camera, view)
        scene.render.filepath = os.path.join(os.path.abspath(args.output), view['imageStem'] + '_preview.jpg')
        bpy.ops.render.render(write_still=True)

        if img:
            bpy.data.images.remove(img)
        if mask:
            bpy.data.images.remove(mask)

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the Blender tests to verify they pass**

Run: `BLENDER_CMD="rez env blender-4 -- blender" rez env meshroom aliceVision pytest -- python -m pytest tests/test_blenderProjection.py tests/test_blenderRender.py -v`
Expected: 8 passed. If `test_blenderProjectionMatchesPinhole` fails, the camera formulas are wrong. Fix `cameraMath` (and its unit tests) to match Blender, never the other way round.

- [ ] **Step 7: Commit**

```bash
git add meshroom/blenderSfmRenderer/scripts/render.py tests/blenderProjectionCheck.py \
        tests/test_blenderProjection.py tests/test_blenderRender.py
git rm --cached meshroom/blenderSfmRenderer/scripts/preview.py
git commit -m "Add Blender render script driven by exported cameras; drop preview.py"
```

---

### Task 4: `ScenePreview` 3.0 node

**Files:**
- Modify (rewrite; currently deleted in working tree): `meshroom/blenderSfmRenderer/ScenePreview.py`
- Modify: `README.md`
- Test: `tests/test_scenePreviewNode.py`

**Interfaces:**
- Consumes: `sfmExport.exportScene` (Task 2); the `render.py` CLI (Task 3).
- Produces: `ScenePreview.preprocess(node)`. It reads `node.cameras.value`, `node.model.value` and `node.internalFolder`, and writes `cameras.json` (plus `landmarks.ply` when `model` doesn't end with `.obj`) into `node.internalFolder`.

- [ ] **Step 1: Write the failing test `tests/test_scenePreviewNode.py`**

```python
import importlib
import sys
import types

import pytest

from conftest import NODE_DIR
from test_sfmExport import makeSfm


@pytest.fixture
def nodeModule():
    '''Import ScenePreview.py as part of its package, as Meshroom does.'''
    sys.path.insert(0, str(NODE_DIR.parent))
    try:
        yield importlib.import_module("blenderSfmRenderer.ScenePreview")
    finally:
        sys.path.remove(str(NODE_DIR.parent))


def fakeNode(tmp_path, cameras, model=""):
    return types.SimpleNamespace(
        cameras=types.SimpleNamespace(value=str(cameras)),
        model=types.SimpleNamespace(value=model),
        internalFolder=str(tmp_path),
    )


def test_nodeDescription(nodeModule):
    nodeDesc = nodeModule.ScenePreview()
    assert nodeModule.__version__ == "3.0"
    assert nodeDesc.hasPreprocess
    assert "{nodeSourceCodeFolder}/scripts/render.py" in nodeDesc.commandLine
    assert "--views {nodeCacheFolder}/cameras.json" in nodeDesc.commandLine
    inputs = {a.name: a for a in nodeDesc.inputs}
    assert "script" not in inputs
    assert inputs["cameras"].commandLineGroup == ""
    outputs = {a.name: a for a in nodeDesc.outputs}
    assert outputs["frames"].semantic == "image"
    assert outputs["frames"].value == "{nodeCacheFolder}/<FILESTEM>_preview.jpg"


def test_preprocessWithMesh(tmp_path, nodeModule):
    sfmPath = makeSfm(tmp_path / "in.sfm", withLandmarks=False)
    nodeModule.ScenePreview().preprocess(fakeNode(tmp_path, sfmPath, model="/data/mesh.obj"))
    assert (tmp_path / "cameras.json").exists()
    assert not (tmp_path / "landmarks.ply").exists()


def test_preprocessWithoutMeshExportsLandmarks(tmp_path, nodeModule):
    sfmPath = makeSfm(tmp_path / "in.sfm")
    nodeModule.ScenePreview().preprocess(fakeNode(tmp_path, sfmPath))
    assert (tmp_path / "cameras.json").exists()
    assert (tmp_path / "landmarks.ply").exists()


def test_preprocessAbcModelFallsBackToLandmarks(tmp_path, nodeModule):
    sfmPath = makeSfm(tmp_path / "in.sfm")
    nodeModule.ScenePreview().preprocess(fakeNode(tmp_path, sfmPath, model="/data/sfm.abc"))
    assert (tmp_path / "landmarks.ply").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `rez env meshroom aliceVision pytest -- python -m pytest tests/test_scenePreviewNode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blenderSfmRenderer.ScenePreview'`.

- [ ] **Step 3: Write `meshroom/blenderSfmRenderer/ScenePreview.py`**

```python
__version__ = "3.0"

from meshroom.core import desc
from pyalicevision import parallelization as avpar

import logging
import os.path

from .sfmExport import exportScene


class ScenePreview(desc.CommandLineNode):
    commandLine = ("{blenderCmdValue} -b --factory-startup --python-exit-code 1 "
                   "--python {nodeSourceCodeFolder}/scripts/render.py -- "
                   "--views {nodeCacheFolder}/cameras.json --landmarks {nodeCacheFolder}/landmarks.ply "
                   "{allParams}")
    size = avpar.DynamicViewsSize("cameras")
    parallelization = desc.Parallelization(blockSize=40)
    commandLineRange = "--rangeStart {rangeStart} --rangeSize {rangeBlockSize}"

    category = "Utils"
    documentation = """
This node uses Blender to render a 3D model from every view of an SfMData.
Each render matches the view's pinhole intrinsics (focal, principal point, pixel ratio) and its pose;
lens distortion is ignored, so renders align with the undistorted images.
If a mesh (.obj) is given it is rendered, otherwise the SfMData landmarks are rendered as a point cloud.
The undistorted views can optionally be used as background.
"""

    inputs = [
        desc.File(
            name="blenderCmd",
            label="Blender Command",
            description="Command to launch Blender.",
            value="blender",
            invalidate=False,
            commandLineGroup="",
        ),
        desc.File(
            name="cameras",
            label="SfMData",
            description="SfMData with the views, poses and intrinsics to use (any format readable by AliceVision).",
            value="",
            commandLineGroup="",
        ),
        desc.File(
            name="model",
            label="Mesh",
            description="Optional mesh (.obj) to render. If empty, the SfMData landmarks are rendered.",
            value="",
        ),
        desc.BoolParam(
            name="useBackground",
            label="Display Background",
            description="Use the undistorted images as background.",
            value=True,
        ),
        desc.File(
            name="undistortedImages",
            label="Undistorted Images",
            description="Folder containing the undistorted images.",
            value="",
            enabled=lambda node: node.useBackground.value,
        ),
        desc.BoolParam(
            name="useMasks",
            label="Apply Masks",
            description="Apply mask to the rendered geometry.",
            value=True,
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Folder containing the masks. If empty, the background image alpha is used.",
            value="",
            enabled=lambda node: node.useMasks.value,
        ),
        desc.GroupAttribute(
            name="pointCloudParams",
            label="Point Cloud Settings",
            commandLineGroup=None,
            enabled=lambda node: not node.model.value.lower().endswith(".obj"),
            description="Settings for landmarks rendering.",
            items=[
                desc.FloatParam(
                    name="particleSize",
                    label="Particle Size",
                    description="Scale of particles used for the point cloud.",
                    value=0.01,
                    range=(0.01, 1.0, 0.01),
                ),
                desc.ChoiceParam(
                    name="particleColor",
                    label="Particle Color",
                    description="Color of particles used for the point cloud.",
                    value="Red",
                    values=["Grey", "White", "Red", "Green", "Magenta"],
                ),
            ],
        ),
        desc.GroupAttribute(
            name="meshParams",
            label="Mesh Settings",
            commandLineGroup=None,
            enabled=lambda node: node.model.value.lower().endswith(".obj"),
            description="Setting for mesh rendering.",
            items=[
                desc.ChoiceParam(
                    name="shading",
                    label="Shading",
                    description="Shading method for visualizing the mesh.",
                    value="wireframe",
                    values=["wireframe", "line_art"],
                ),
                desc.ChoiceParam(
                    name="edgeColor",
                    label="Edge Color",
                    description="Color of the mesh edges.",
                    value="Red",
                    values=["Grey", "White", "Red", "Green", "Magenta"],
                ),
            ],
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output",
            description="Output folder.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="frames",
            label="Frames",
            description="Frames rendered in Blender.",
            semantic="image",
            value="{nodeCacheFolder}/<FILESTEM>_preview.jpg",
            commandLineGroup="",
        ),
    ]

    def preprocess(self, node):
        '''Export cameras (and landmarks when no mesh is given) once, before the Blender chunks.'''
        model = node.model.value
        useMesh = model.lower().endswith(".obj")
        if model and not useMesh:
            logging.warning(f"Model '{model}' is not an .obj mesh: rendering the SfMData landmarks instead.")

        folder = node.internalFolder
        exportScene(
            node.cameras.value,
            os.path.join(folder, "cameras.json"),
            None if useMesh else os.path.join(folder, "landmarks.ply"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `rez env meshroom aliceVision pytest -- python -m pytest tests/test_scenePreviewNode.py -v`
Expected: 4 passed.

- [ ] **Step 5: Update `README.md`**

Replace the whole file with:

````markdown
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
````

- [ ] **Step 6: Run the full test suite**

Run: `BLENDER_CMD="rez env blender-4 -- blender" rez env meshroom aliceVision pytest -- python -m pytest tests -v`
Expected: 23 passed (6 cameraMath, 5 sfmExport, 4 projection, 4 render, 4 node; parametrized cases count individually), 0 skipped.

- [ ] **Step 7: Commit**

```bash
git add meshroom/blenderSfmRenderer/ScenePreview.py README.md tests/test_scenePreviewNode.py
git commit -m "ScenePreview 3.0: export with pyalicevision in preprocess, render in Blender"
```

---

### Task 5: Smoke run in Meshroom on a real dataset

**Files:** none (verification only).

- [ ] **Step 1: Build and load the plugin**

Build the rez package from `/s/apps/users/servantf/packages/mrBlenderSfmRenderer/servantf` as usual for this package (`rez build --install` or the team's equivalent). Then start Meshroom with the plugin.

- [ ] **Step 2: Render a reconstructed scene**

In a graph with a finished `StructureFromMotion` → `PrepareDenseScene` (undistorted images) → `Texturing` (or `Meshing`):
- Add a `ScenePreview` node. Connect `cameras` to the SfM output and `undistortedImages` to `PrepareDenseScene.output`, and leave `model` empty. Compute.
- Duplicate the node, connect `model` to the mesh `.obj`, and compute.

- [ ] **Step 3: Check by eye in the 2D viewer** (`frames` output)

- Landmark render: points sit on the image features they come from, across several views, including views near the image borders.
- Mesh render: the wireframe outline follows the object's silhouette in the background image.
- The node log shows `preprocess.log` with `Exported N/M renderable views` and one log per chunk.

If points are offset by a constant amount, the world convention for the mesh or the PLY is wrong (spec: everything stays in AliceVision world coordinates). Report it with a screenshot rather than adding an ad-hoc rotation.

- [ ] **Step 4: Note the result** in the PR description (dataset used, views checked).
