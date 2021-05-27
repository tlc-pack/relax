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

VOID_ARG_ = 0xFE0321975A
    
class VMFuncScope(object):
    stack = []
    def __init__(self, func_name, num_inputs, callback):
        self.func_name = func_name
        self.num_inputs = num_inputs
        self.callback = callback

    def __enter__(self):
        VMFuncScope.stack.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        if not self.callback():
            raise ValueError("an unexpected register is used as input")
        VMFuncScope.stack.pop()

@tvm._ffi.register_object("relax.Builder")
class Builder(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.BuilderCreate)

    def r(self, idx):
        return _ffi_api.BuilderR(self, idx)

    def imm(self, value):
        return _ffi_api.BuilderImm(self, value)

    def c(self, idx):
        return _ffi_api.BuilderC(self, idx)

    def function(self, func_name, num_inputs=0):
        """set register file here"""
        def callback():
            return _ffi_api.BuilderCheck(self)
        _ffi_api.BuilderFunction(self, func_name, num_inputs)
        return VMFuncScope(func_name, num_inputs, callback) 

    def _check_scope(self):
        if len(VMFuncScope.stack) == 0:
            raise ValueError("emit should happen in a function scope")

    def emit_constant(self, const):
        return _ffi_api.BuilderEmitConstant(self, const)

    def emit_call(self, name, args=[], dst=None):
        self._check_scope()
        if dst is None:
            dst = VOID_ARG_
        args_ = []
        for arg in args:
            if isinstance(arg, tvm.nd.NDArray):
                new_arg = self.emit_constant(arg)
                args_.append(new_arg)
            else:
                args_.append(arg)
        _ffi_api.BuilderEmitCall(self, name, args_, dst)

    def emit_ret(self, result):
        self._check_scope()
        _ffi_api.BuilderEmitRet(self, result)

    def get(self):
        """formalize and return the executable"""
        return _ffi_api.BuilderGet(self)

