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
import tvm
from tvm.runtime import Object
from tvm._ffi.base import _LIB, check_call
from . import _ffi_api

class ArgKind(IntEnum):
    REGISTER = 0
    IMMEDIATE = 1
    CONSTIDX = 2

VOID_ARG_ = 0xFE0321975A
    
def _create_arg(kind, value):
    return (int(kind) << 56) | (value & ((1 << 56) - 1))

class VMFuncScope(object):
    stack = []
    def __init__(self, func_name, num_inputs):
        self.func_name = func_name
        self.num_inputs = num_inputs

    def __enter__(self):
        VMFuncScope.stack.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        VMFuncScope.stack.pop()


@tvm._ffi.register_object("relax.Builder")
class Builder(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.BuilderCreate)

    def r(self, idx):
        return _create_arg(ArgKind.REGISTER, idx)

    def imm(self, value):
        return _create_arg(ArgKind.IMMEDIATE, value)

    def c(self, idx):
        return _create_arg(ArgKind.CONSTIDX, idx)

    def function(self, func_name, num_inputs=0):
        """set register file here"""
        _ffi_api.BuilderFunction(self, func_name, num_inputs)
        return VMFuncScope(func_name, num_inputs) 

    def _check_scope(self):
        if len(VMFuncScope.stack) == 0:
            raise ValueError("emit should happen in a function scope")

    def emit_call(self, name, args=[], ret=None):
        self._check_scope()
        if ret is None:
            ret = VOID_ARG_
        args_ = []
        for arg in args:
            if isinstance(arg, tvm.nd.NDArray):
                new_arg = _ffi_api.BuilderEmitConstant(self, arg)
                args_.append(new_arg)
            else:
                args_.append(arg)
        _ffi_api.BuilderEmitCall(self, name, args_, ret)

    def emit_ret(self, result):
        self._check_scope()
        _ffi_api.BuilderEmitRet(self, result)

    def get(self):
        """formalize and return the executable"""
        return _ffi_api.BuilderGet(self)

