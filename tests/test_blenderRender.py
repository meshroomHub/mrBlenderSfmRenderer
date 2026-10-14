import json
import struct
import zlib

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


def writePng(path, width, height, rgb):
    '''Write a solid-color 8-bit RGB PNG.'''
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    row = b"\x00" + bytes(rgb) * width
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(row * height)))
        f.write(chunk(b"IEND", b""))


def readPixels(path):
    '''Read an 8-bit RGB image as a numpy array (height, width, 3).'''
    from pyalicevision import image
    img = image.Image_RGBColor()
    image.readImage(str(path), img, image.ImageReadOptions(image.EImageColorSpace_NO_CONVERSION))
    return np.asarray(img.getNumpyArray(), dtype=float)


def test_renderBackgroundColorPreserved(tmp_path, blenderCmd):
    '''The undistorted background must not be tone-mapped (Blender 4 defaults to AgX).'''
    writeScene(tmp_path)
    background = tmp_path / "undistorted"
    background.mkdir()
    writePng(background / "imgA.png", 320, 240, (0, 0, 200))
    writePng(background / "imgB.png", 200, 300, (0, 0, 200))
    out = tmp_path / "out"
    out.mkdir()
    args = baseArgs(tmp_path, out)
    args[args.index("--useBackground") + 1] = "True"
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py", args + ["--undistortedImages", background])
    corner = readPixels(out / "imgA_preview.jpg")[5, 5]
    assert np.abs(corner - [0, 0, 200]).max() <= 8, corner


def writeUprightScene(folder):
    '''One camera at AliceVision world (0, 0, -5) looking +z, identity rotation.

    Geometry is placed around world (0, -1, 0): AliceVision y points down in the image,
    so it must appear in the upper half of the render.
    '''
    view = {"viewId": "1", "imageStem": "up", "valid": True, "width": 200, "height": 200,
            "fx": 300.0, "fy": 300.0, "ox": 0.0, "oy": 0.0, "R": IDENTITY, "C": [0.0, 0.0, -5.0]}
    (folder / "cameras.json").write_text(json.dumps({"views": [view]}))


def brightRows(path):
    '''Row indices containing at least one bright pixel.'''
    pixels = readPixels(path)
    return np.nonzero(pixels.max(axis=(1, 2)) > 60)[0]


def test_meshInAliceVisionConventionRendersUpright(tmp_path, blenderCmd):
    '''AliceVision Mesh::save writes vertices as (x, -y, -z): the render must undo it.'''
    writeUprightScene(tmp_path)
    # Triangle around world (0, -1, 0), written like AliceVision does (y and z negated), both windings
    triangle = [(-0.3, -1.2, 0.0), (0.3, -1.2, 0.0), (0.0, -0.8, 0.0)]
    lines = ["v %f %f %f" % (x, -y, -z) for x, y, z in triangle] + ["f 1 2 3", "f 1 3 2"]
    (tmp_path / "mesh.obj").write_text("\n".join(lines) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py",
               ["--views", tmp_path / "cameras.json", "--model", tmp_path / "mesh.obj", "--output", out,
                "--useBackground", "False", "--useMasks", "False", "--shading", "wireframe", "--edgeColor", "Green"])
    rows = brightRows(out / "up_preview.jpg")
    assert len(rows) > 0, "mesh not visible"
    assert rows.max() < 100, f"mesh drawn in lower half (rows {rows.min()}..{rows.max()})"


def test_landmarksRenderUpright(tmp_path, blenderCmd):
    '''Landmarks come from the sfmData world unchanged.'''
    writeUprightScene(tmp_path)
    points = np.zeros(5, dtype=PLY_DTYPE)
    points[:] = [(dx, -1.0 + dy, 0.0, 255, 0, 0) for dx, dy in [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]]
    with open(tmp_path / "landmarks.ply", "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 5\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        points.tofile(f)
    out = tmp_path / "out"
    out.mkdir()
    runBlender(blenderCmd, SCRIPTS_DIR / "render.py",
               ["--views", tmp_path / "cameras.json", "--landmarks", tmp_path / "landmarks.ply", "--model", "",
                "--output", out, "--useBackground", "False", "--useMasks", "False",
                "--particleSize", "0.02", "--particleColor", "White"])
    rows = brightRows(out / "up_preview.jpg")
    assert len(rows) > 0, "landmarks not visible"
    assert rows.max() < 100, f"landmarks drawn in lower half (rows {rows.min()}..{rows.max()})"
