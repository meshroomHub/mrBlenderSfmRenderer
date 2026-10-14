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
