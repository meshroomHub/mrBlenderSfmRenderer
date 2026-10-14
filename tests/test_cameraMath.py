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
