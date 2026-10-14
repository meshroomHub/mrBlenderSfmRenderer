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
