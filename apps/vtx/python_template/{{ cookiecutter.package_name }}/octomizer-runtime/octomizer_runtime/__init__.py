import os
import tvm
import tvm.runtime
from tvm import relax
from pathlib import Path
from typing import Any

PKG_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


class OctomizedModelVM:
    """
    Wrapper around TVM machinery for executing your model
    """

    _vm: Any
    _ctx: Any
    runtime_module: Any

    def __init__(
        self, exported_module_path=None, ctx=None
    ):
        exported_module_path = exported_module_path or PKG_PATH.joinpath("module.so")

        loaded_lib = tvm.runtime.load_module(str(exported_module_path))
        self._vm = relax.VirtualMachine(loaded_lib, ctx)
        self._ctx = ctx

    def __call__(self, *args, **kwargs):
        return self._vm["main"](*args, **kwargs)

    def measure_runtime(self, *args, number=100, repeat=10):
        # Measure runtime
        ftimer = self._vm.module.time_evaluator(
            func_name="main", dev=self._ctx, number=number, repeat=repeat
        )
        result = ftimer(*args)
        print(result)
