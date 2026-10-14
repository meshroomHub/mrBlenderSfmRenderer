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
    # Keep background images and colors unchanged (Blender 4 defaults to AgX tone mapping)
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'


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
        # AliceVision meshes are saved as (x, -y, -z): bring them back to the sfmData world
        obj.matrix_world = mathutils.Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
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
