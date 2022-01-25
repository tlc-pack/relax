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
# pylint: disable=invalid-name
"""Relax transformation passes."""
import tvm.ir
from . import _ffi_api


@tvm._ffi.register_object("relax.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.Function in a module. A function
    pass class should be created through `function_pass`.
    """


def FMARewrite() -> tvm.ir.transform.Pass:
    """Perform fused multiply add rewriting in dataflow blocks.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.FMARewrite()


def ToNonDataflow() -> tvm.ir.transform.Pass:
    """Transform all dataflow structure to non-dataflow version.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ToNonDataflow()


def CallTIRRewrite() -> tvm.ir.transform.Pass:
    """Perform explicit tensor allocation for call_tir.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.CallTIRRewrite()


def VMMemoryLower() -> tvm.ir.transform.Pass:
    """Perform memory lowering. Lowers the relax.builtin.alloc_tensor intrinsic to VM intrinsics.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.VMMemoryLower()


def VMShapeLower() -> tvm.ir.transform.Pass:
    """Lower the shape expressions in relax to VM shape heap manipulations and generate related
    TIR functions to do shape calculations.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.VMShapeLower()


def ToANF() -> tvm.ir.transform.Pass:
    """Transforming Relax IR to A-normal form.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ToANF()


def ResolveGlobals() -> tvm.ir.transform.Pass:
    """Resolve global variables using string equality. This ensures all GlobalVars in the IR refer
    to the correct GlobalVar of the input IRModule. An error is reported if any GlobalVar cannot be
    resolved.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ResolveGlobals()
