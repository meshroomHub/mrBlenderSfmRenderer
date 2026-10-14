import importlib
import sys
import types

import pytest

from conftest import NODE_DIR
from test_sfmExport import makeSfm


@pytest.fixture
def nodeModule():
    '''Import ScenePreview.py as part of its package, as Meshroom does.'''
    sys.path.insert(0, str(NODE_DIR.parent))
    try:
        yield importlib.import_module("blenderSfmRenderer.ScenePreview")
    finally:
        sys.path.remove(str(NODE_DIR.parent))


def fakeNode(tmp_path, cameras, model=""):
    return types.SimpleNamespace(
        cameras=types.SimpleNamespace(value=str(cameras)),
        model=types.SimpleNamespace(value=model),
        internalFolder=str(tmp_path),
    )


def test_nodeDescription(nodeModule):
    nodeDesc = nodeModule.ScenePreview()
    assert nodeModule.__version__ == "3.0"
    assert nodeDesc.hasPreprocess
    assert "{nodeSourceCodeFolder}/scripts/render.py" in nodeDesc.commandLine
    assert "--views {nodeCacheFolder}/cameras.json" in nodeDesc.commandLine
    inputs = {a.name: a for a in nodeDesc.inputs}
    assert "script" not in inputs
    assert inputs["cameras"].commandLineGroup == ""
    outputs = {a.name: a for a in nodeDesc.outputs}
    assert outputs["frames"].semantic == "image"
    assert outputs["frames"].value == "{nodeCacheFolder}/<FILESTEM>_preview.jpg"


def test_preprocessWithMesh(tmp_path, nodeModule):
    sfmPath = makeSfm(tmp_path / "in.sfm", withLandmarks=False)
    nodeModule.ScenePreview().preprocess(fakeNode(tmp_path, sfmPath, model="/data/mesh.obj"))
    assert (tmp_path / "cameras.json").exists()
    assert not (tmp_path / "landmarks.ply").exists()


def test_preprocessWithoutMeshExportsLandmarks(tmp_path, nodeModule):
    sfmPath = makeSfm(tmp_path / "in.sfm")
    nodeModule.ScenePreview().preprocess(fakeNode(tmp_path, sfmPath))
    assert (tmp_path / "cameras.json").exists()
    assert (tmp_path / "landmarks.ply").exists()


def test_preprocessAbcModelFallsBackToLandmarks(tmp_path, nodeModule):
    sfmPath = makeSfm(tmp_path / "in.sfm")
    nodeModule.ScenePreview().preprocess(fakeNode(tmp_path, sfmPath, model="/data/sfm.abc"))
    assert (tmp_path / "landmarks.ply").exists()
