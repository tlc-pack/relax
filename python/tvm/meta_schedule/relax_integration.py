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
from typing import Any, List, Union, Dict, Optional

import tvm
from tvm._ffi import get_global_func
from tvm.ir import IRModule
from tvm.meta_schedule import ExtractedTask
from tvm.target import Target
from tvm.relax.expr import Function as RelaxFunc
from tvm.runtime import NDArray


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

    extract_task_func = get_global_func(
        "relax.backend.MetaScheduleExtractTask",
        allow_missing=False,
    )

    if isinstance(mod, RelaxFunc):
        mod = IRModule.from_expr(mod)

    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {}

    if params:
        mod = tvm.relax.transform.BindParams("main", params)(mod)

    with target, tvm.transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
    ):
        return list(extract_task_func(mod, target))
