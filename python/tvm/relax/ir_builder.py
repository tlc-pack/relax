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
"""Developer API of constructing Relax AST."""
from typing import List, Optional, Union, Dict
from tvm.relay.expr import Tuple
from tvm.runtime import Object
from .expr import *
from tvm._ffi.base import _LIB, check_call
from . import _ffi_api


@tvm._ffi.register_object("relax.FunctionScope")
class FunctionScope(Object):
    """Auxiliary scope for function"""

    def __init__(self, irbuilder):
        self.__init_handle_by_constructor__(_ffi_api.CreateFunctionScope, irbuilder)

    def __enter__(self):
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_api.ExitFunctionScope(self)


@tvm._ffi.register_object("relax.DataflowScope")
class DataflowScope(Object):
    """Auxiliary scope for Dataflow block"""

    def __init__(self, irbuilder):
        self.__init_handle_by_constructor__(_ffi_api.CreateDataflowScope, irbuilder)

    def __enter__(self):
        _ffi_api.EnterDataflowScope(self)

    def __exit__(self, ptype, value, trace):
        _ffi_api.ExitDataflowScope(self)


@tvm._ffi.register_object("relax.IRBuilder")
class IRBuilder(Object):
    """A builder to build Relax IR for testing and dev.

    Examples
    --------
    .. code-block:: python

        m = tir.Var("m", "int32")
        n = tir.Var("n", "int32")
        dtype0 = rx.DynTensorType(rank=2, dtype="float16")
        dtype1 = rx.DynTensorType(rank=1, dtype="float16")
        x = rx.Var("x", [m, n], dtype0)
        y = rx.Var("y", [n], dtype1)
        ib = rx.IRBuilder()
        with ib.function("func", [x, y]):
            with ib.dataflow() as df:
                lv0 = ib.emit(rx.add(x, y))
                lv1 = ib.emit(rx.multiply(lv0, y))
                gv0 = ib.emit_df_output(lv1)
            ib.emit_output(gv0)
        func = ib.get()
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.IRBuilderCreate)

    def function(self,
                 name: str,
                 params: Union[Var, Tuple, List[Var]]) -> FunctionScope:
        """Annotate a Relax function.

        Parameters
        ----------
        name : str
            The name of the function

        params : tvm.relax.Var or List[tvm.relax.Var]
            The parameters of the function
        """
        if not isinstance(params, (list, tuple)):
            params = [params]

        _ffi_api.IRBuilderFillFuncNameParam(self, name, params)
        return FunctionScope(self)

    def dataflow(self) -> DataflowScope:
        """Annotate a Relax dataflow block."""
        return DataflowScope(self)

    def emit(self,
             call: relay.Call) -> Var:
        """Emit a call node.
        This infers the shape and type of the CallNode, create a variable,
        and bind the CallNode to the variable.

        Parameters
        ----------
        call : tvm.relay.Call
            The Call to be emitted

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets binded to the call
        """
        return _ffi_api.IRBuilderEmit(self, call)

    def emit_df_output(self,
                       var: Var) -> Var:
        """Emit a dataflow block's output variable, and it can be used outside the dataflow block.

        Parameters
        ----------
        var : tvm.relax.Var
            The output variable of the current dataflow block

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets binded to the output var
        """
        return _ffi_api.IRBuilderEmitDataflowOutput(self, var)

    def emit_output(self,
                    output: Union[Expr, Tuple, List[Expr]]) -> None:
        """Emit function outputs.

        Parameters
        ----------
        output : Expr | List[Expr]
            The output variable(s) of the current function
        """
        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        _ffi_api.IRBuilderEmitOutput(self, output)

    def get(self) -> Function:
        """Return the function being built."""
        return _ffi_api.IRBuilderGet(self)
