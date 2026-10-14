__version__ = "3.0"

from meshroom.core import desc
from pyalicevision import parallelization as avpar

import logging
import os.path

from .sfmExport import exportScene


class ScenePreview(desc.CommandLineNode):
    # Paths outside {allParams} are not quoted by Meshroom
    commandLine = ('{blenderCmdValue} -b --factory-startup --python-exit-code 1 '
                   '--python "{nodeSourceCodeFolder}/scripts/render.py" -- '
                   '--views "{nodeCacheFolder}/cameras.json" --landmarks "{nodeCacheFolder}/landmarks.ply" '
                   '{allParams}')
    size = avpar.DynamicViewsSize("cameras")
    parallelization = desc.Parallelization(blockSize=40)
    commandLineRange = "--rangeStart {rangeStart} --rangeSize {rangeBlockSize}"

    category = "Utils"
    documentation = """
This node uses Blender to render a 3D model from every view of an SfMData.
Each render matches the view's pinhole intrinsics (focal, principal point, pixel ratio) and its pose;
The SfMData views are expected to be undistorted: lens distortion is ignored.
If a mesh (.obj) is given it is rendered, otherwise the SfMData landmarks are rendered as a point cloud.
Each render is composited over its view image.
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
