# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Meta schedule integration with high-level IR"""
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union
import tvm

# from tvm._ffi import register_object
from tvm.ir import IRModule, transform
from tvm.runtime import NDArray, Object
from tvm.target import Target
from tvm.tir import PrimFunc

# from tvm.relax.meta_schedule import _ffi_api
from tvm.relax import vm
from tvm.relax.expr import Function as RelaxFunc
from tvm.relax.ty import DynTensorType
from tvm.meta_schedule.database import Database
from tvm.meta_schedule.integration import ExtractedTask, TaskExtraction, MetaScheduleContext


# Simply extracts tir PrimFuncs from the input IRModule
def _base_partitioner(mod: tvm.IRModule) -> List[tvm.IRModule]:
    partitions = []
    for gv in mod.get_global_vars():
        if isinstance(mod[gv], PrimFunc):
            tir_mod = IRModule({})
            tir_mod[gv] = mod[gv]
            partitions.append(tir_mod)
    return partitions


def extract_task_from_relax(
    mod: Union[IRModule, RelaxFunc],
    target: Target,
    params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Dict[str, DynTensorType] = {},
    disabled_pass: List[str] = [],
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relax program.

    Parameters
    ----------
    mod : Union[tvm.IRModule, tvm.relax.Function]
        The module or function to tune
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    opt_level : int
        The optimization level of the compiler
    pass_config : Dict[str, DynTensorType]
        The pass config of the compiler
    disabled_pass : List[str]
        The list of disabled passes of the compiler

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this network
    """

    @contextmanager
    def _autotvm_silencer():
        from tvm import autotvm  # pylint: disable=import-outside-toplevel

        silent = autotvm.GLOBAL_SCOPE.silent
        autotvm.GLOBAL_SCOPE.silent = True
        try:
            yield
        finally:
            autotvm.GLOBAL_SCOPE.silent = silent

    def _thread_run(func: Callable[[], None]) -> None:
        import threading  # pylint: disable=import-outside-toplevel

        thread = threading.Thread(target=func)
        thread.start()
        thread.join()

    env = TaskExtraction()
    if isinstance(mod, RelaxFunc):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)

    def _func():
        with env, _autotvm_silencer(), transform.PassContext(
            config=pass_config,
            disabled_pass=disabled_pass,
            opt_level=opt_level,
        ):
            tir_partitions = _base_partitioner(mod)
            for i, tir_mod in enumerate(tir_partitions):
                func_name = tir_mod.get_global_vars()[0].name_hint
                MetaScheduleContext.query_inside_with_scope(func_name, tir_mod, target, [tir_mod])

    _thread_run(_func)
    return env.tasks
