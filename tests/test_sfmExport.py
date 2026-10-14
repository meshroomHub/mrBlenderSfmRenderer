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
    assert views[0] == {"viewId": "5", "imageStem": "IMG_0005", "imagePath": "/data/IMG_0005.jpg", "valid": False}

    v = views[1]
    assert v["valid"] is True
    assert v["imageStem"] == "IMG_0010"
    assert v["imagePath"] == "/data/IMG_0010.jpg"
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
