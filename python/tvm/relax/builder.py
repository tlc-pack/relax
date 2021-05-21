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
    
def create_arg(kind, value):
    return (int(kind) << 56) | (value & ((1 << 56) - 1))

@tvm._ffi.register_object("relax.Executable")
class Executable(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Executable)

    def astext(self):
        return _ffi_api.ExecutableAsText(self) 
        

@tvm._ffi.register_object("relax.Builder")
class Builder(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.BuilderCreate)

    def r(self, idx):
        return create_arg(ArgKind.REGISTER, idx)

    def imm(self, value):
        return create_arg(ArgKind.IMMEDIATE, value)

    def const(self, idx):
        return create_arg(ArgKind.CONSTIDX, idx)

    def emit_call(self, name, args=[], ret=None):
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

    def get(self):
        return _ffi_api.BuilderGet(self)

