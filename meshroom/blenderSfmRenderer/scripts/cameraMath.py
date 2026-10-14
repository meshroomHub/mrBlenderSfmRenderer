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
