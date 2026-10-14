'''Shared test helpers.

Run from the repo root:
    rez env meshroom aliceVision pytest -- python -m pytest tests -v
Blender tests also need:
    export BLENDER_CMD="rez env blender-4 -- blender"
'''
import os
import pathlib
import shlex
import struct
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
NODE_DIR = REPO / "meshroom" / "blenderSfmRenderer"
SCRIPTS_DIR = NODE_DIR / "scripts"

# Import helper modules directly (the package name would clash with the real meshroom package).
for path in (str(NODE_DIR), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def blenderCmd():
    '''Command used to launch Blender, from the BLENDER_CMD environment variable.'''
    cmd = os.environ.get("BLENDER_CMD", "")
    if not cmd:
        pytest.skip("BLENDER_CMD is not set")
    return shlex.split(cmd)


def runBlender(blenderCmd, script, args):
    '''Run a Python script in background Blender and fail with its output on error.'''
    cmd = blenderCmd + ["-b", "--factory-startup", "--python-exit-code", "1",
                        "--python", str(script), "--"] + [str(a) for a in args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Blender failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return proc


def jpegSize(path):
    '''Return (width, height) read from the SOF marker of a JPEG file.'''
    with open(path, "rb") as f:
        data = f.read()
    assert data[:2] == b"\xff\xd8", f"{path} is not a JPEG"
    i = 2
    while i < len(data):
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    raise ValueError(f"No SOF marker in {path}")
