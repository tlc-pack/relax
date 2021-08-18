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
"""Developer API of building Relax IR AST."""

from tvm.relay.expr import Tuple
from tvm.runtime import Object
from .expr import *
from tvm._ffi.base import _LIB, check_call
from . import _ffi_api


class FunctionScope(object):
    """Auxiliary scope for function"""

    def __init__(self, name, params, build_block, build_function):
        self.name = name
        self.params = params
        self.build_block = build_block
        self.build_function = build_function

    def __enter__(self):
        return self

    def __exit__(self, ptype, value, trace):
        self.build_block()
        self.build_function(self.name, self.params)


class DataflowScope(object):
    """Auxiliary scope for Dataflow block"""

    def __init__(self, build_block, flip_state):
        self.build_block = build_block
        self.flip_state = flip_state

    def __enter__(self):
        self.build_block()
        self.flip_state()

    def __exit__(self, ptype, value, trace):
        self.build_block()
        self.flip_state()


@tvm._ffi.register_object("relax.IRBuilder")
class IRBuilder(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.IRBuilderCreate)

    def function(self, name, params):
        if not isinstance(params, (list, tuple)):
            params = [params]

        def build_block():
            return _ffi_api.IRBuilderBuildBlock(self)

        def build_function(name, params):
            return _ffi_api.IRBuilderBuildFunction(self, name, params)

        return FunctionScope(name, params, build_block, build_function)

    def dataflow(self):
        def build_block():
            return _ffi_api.IRBuilderBuildBlock(self)

        def flip_state():
            return _ffi_api.IRBuilderFlipState(self)

        return DataflowScope(build_block, flip_state)

    def emit(self, call):
        return _ffi_api.IRBuilderEmit(self, call)

    def emit_df_output(self, var):
        return _ffi_api.IRBuilderEmitDataflowOutput(self, var)

    def emit_output(self, output):
        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        _ffi_api.IRBuilderEmitOutput(self, output)

    def get(self):
        return _ffi_api.IRBuilderGet(self)
