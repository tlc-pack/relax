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
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
import tvm.ir
from . import _ffi_api

@tvm._ffi.register_object("relax.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.Function in a module. A function
    pass class should be created through `function_pass`.
    """

def FMARewrite() -> tvm.transform.Pass:
    """Perform fused multiply add rewriting in dataflow blocks.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.FMARewrite()


def ToNonDataflow() -> tvm.transform.Pass:
    """Transform all dataflow structure to non-dataflow version.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.ToNonDataflow()


def CallDPSRewrite() -> tvm.transform.Pass:
    """Perform explicit tensor allocation for call_dps.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.CallDPSRewrite()


def VMMemoryLower() -> tvm.transform.Pass:
    """Perform memory lowering. Lowers the relax.builtin.alloc_tensor intrinsic to VM intrinsics.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.VMMemoryLower()


def VMShapeLower() -> tvm.transform.Pass:
    """Lower the shape expressions in relax to VM shape heap manipulations and generate related 
    TIR functions to do shape calculations.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.VMShapeLower()


def ToANF() -> tvm.transform.Pass:
    """Transforming Relax IR to A-normal form.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.ToANF()
