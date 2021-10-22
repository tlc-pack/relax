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

from enum import IntEnum
from typing import Optional, Union, List
import tvm
from tvm._ffi._ctypes.packed_func import TVMRetValueHandle
from tvm.runtime import Object
from tvm.runtime.container import ShapeTuple
from tvm._ffi.base import _LIB, check_call
from . vm import Executable
from . import _ffi_api

class SpecialReg(IntEnum):
    """Magic numbers that represent special registers in vm."""
    VOID_ARG = 0x00EC66FE0321975A
    VM_STATE = 0x008D14FA4379015C

class VMFuncScope(object):
    """An object corresponds to each VM function, working as a context manager."""
    stack = []

    def __enter__(self):
        VMFuncScope.stack.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        VMFuncScope.stack.pop()

@tvm._ffi.register_object("relax.ExecBuilder")
class ExecBuilder(Object):
    """A builder to emit instructions and build executable for the virtual machine."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ExecBuilderCreate)

    def r(self, idx: int) -> int:
        """set instruction's argument as a register."""
        return _ffi_api.ExecBuilderR(self, idx)

    def imm(self, value: int) -> int:
        """set instruction's argument as an immediate."""
        return _ffi_api.ExecBuilderImm(self, value)

    def c(self, idx: int) -> int:
        """set instruction's argument as a constant."""
        return _ffi_api.ExecBuilderC(self, idx)

    def void_arg(self) -> int:
        return self.r(SpecialReg.VOID_ARG)

    def vm_state(self) -> int:
        return self.r(SpecialReg.VM_STATE)

    def function(self, func_name: str, num_inputs: Optional[int] = 0) -> VMFuncScope:
        """annotate a VM function."""
        _ffi_api.ExecBuilderFunction(self, func_name, num_inputs)
        return VMFuncScope()

    def _check_scope(self) -> None:
        if len(VMFuncScope.stack) == 0:
            raise ValueError("emit should happen in a function scope")

    def emit_constant(self, const: TVMRetValueHandle) -> int:
        return _ffi_api.ExecBuilderEmitConstant(self, const)

    def emit_call(
        self,
        name: str,
        args: Optional[List[Union[tvm.nd.NDArray, tvm.DataType]]] = [],
        dst: int = None,
    ) -> None:
        """emit a call instruction which calls a packed function."""
        self._check_scope()
        if dst is None:
            dst = SpecialReg.VOID_ARG
        args_ = []
        for arg in args:
            if isinstance(arg, tuple):
                shape_tuple = ShapeTuple(arg)
                new_arg = self.emit_constant(shape_tuple)
                args_.append(new_arg)
            elif isinstance(arg, (tvm.nd.NDArray, tvm.DataType, ShapeTuple)):
                new_arg = self.emit_constant(arg)
                args_.append(new_arg)
            else:
                args_.append(arg)
        _ffi_api.ExecBuilderEmitCall(self, name, args_, dst)

    def emit_ret(self, result: int) -> None:
        """emit a return instruction"""
        self._check_scope()
        _ffi_api.ExecBuilderEmitRet(self, result)

    def get(self) -> Executable:
        """return the executable"""
        return _ffi_api.ExecBuilderGet(self)
