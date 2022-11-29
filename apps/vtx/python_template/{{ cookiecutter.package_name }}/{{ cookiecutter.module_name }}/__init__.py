from pathlib import Path
import os
import tvm

import octomizer_runtime

PKG_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def model(
    exported_module_path=None,
    ctx=None,
):
    exported_module_path = exported_module_path or PKG_PATH.joinpath("packaged.so")

    if ctx is None:
        ctx = {{cookiecutter.ctx}}

    return octomizer_runtime.OctomizedModelVM(
        exported_module_path, ctx
    )
