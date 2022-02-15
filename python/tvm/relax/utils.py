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
"""Utility functions for Relax"""
from typing import List
from tvm.tir import PrimFunc
from tvm import IRModule

# Simply extracts tir PrimFuncs from the input IRModule
def tir_partitioner(mod: IRModule) -> List[IRModule]:
    partitions = []
    for gvar in mod.get_global_vars():
        if isinstance(mod[gvar], PrimFunc):
            tir_mod = IRModule({})
            tir_mod[gvar] = mod[gvar]
            partitions.append(tir_mod)
    return partitions
