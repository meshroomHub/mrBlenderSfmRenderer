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
