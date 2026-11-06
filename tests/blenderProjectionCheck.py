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
