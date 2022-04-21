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
from typing import Any, List, Union, Tuple, Dict, Optional

import tvm
from tvm.ir import IRModule, structural_hash, structural_equal
from tvm.meta_schedule import ExtractedTask
from tvm.target import Target
from tvm.relax.expr import Function as RelaxFunc
from tvm.relax.utils import tir_partitioner
from tvm.runtime import NDArray


def deduplicate_extracted_tasks(
    mods: List[IRModule],
) -> Tuple[List[IRModule], List[int]]:
    """Remove duplicate modules.
    Parameters
    ----------
    mods : List[IRModule]
        The list of IRModule.
    Returns
    -------
    tasks : Tuple[List[IRModule], List[int]]
        A tuple containing the deduplicated modules and the count for each module.
    """
    hash2modules: Dict[int, List[IRModule]] = {}
    hash2counts: Dict[int, List[int]] = {}
    for mod in mods:
        shash = structural_hash(mod)
        if shash in hash2modules:
            is_dup = False
            for i, relax_mod in enumerate(hash2modules[shash]):
                # duplicate module was found
                if structural_equal(mod, relax_mod):
                    hash2counts[shash][i] += 1
                    is_dup = True
                    break
            if is_dup is False:
                # hash conflict but actually different modules
                hash2modules[shash].append(mod)
                hash2counts[shash].append(1)

        else:
            hash2modules[shash] = [mod]
            hash2counts[shash] = [1]

    dedup: List[IRModule] = []
    count: List[int] = []
    for shash, relax_mods in hash2modules.items():
        for i, mod in enumerate(relax_mods):
            dedup.append(mod)
            count.append(hash2counts[shash][i])
    return dedup, count


def extract_task_from_relax(
    mod: Union[IRModule, RelaxFunc],
    target: Target,
    params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Optional[Dict[str, Any]] = None,
    disabled_pass: Optional[List[str]] = None,
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relax program.

    Parameters
    ----------
    mod : Union[tvm.IRModule, tvm.relax.Function]
        The module or function to tune
    target : tvm.target.Target
        The compilation target

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this module
    """
    if isinstance(mod, RelaxFunc):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)

    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {}

    if params:
        mod = tvm.relax.transform.BindParams("main", params)(mod)

    tir_partitions = tir_partitioner(mod)
    tir_mods, tir_counts = deduplicate_extracted_tasks(tir_partitions)
    tasks = []
    with target, tvm.transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
    ):
        for i, tir_mod in enumerate(tir_mods):
            task_name = tir_mod.get_global_vars()[0].name_hint
            # The second arg to ExtractedTask is supposed to be a high-level IRModule,
            # passing tir_mod as a workaround.
            tasks.append(ExtractedTask(task_name, tir_mod, target, [tir_mod], tir_counts[i]))
    return tasks
