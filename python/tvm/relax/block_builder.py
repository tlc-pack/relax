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
from tvm import relax as rx
from .expr import *
from tvm._ffi.base import _LIB, check_call
from . import _ffi_api


class FunctionScope(object):
    """Auxiliary scope for function"""

    def __init__(self, irbuilder):
        self._ib = irbuilder

    def __enter__(self):
        _ffi_api.BlockBuilderBeginBindingBlock(self._ib)

    def __exit__(self, ptype, value, trace):
        block = _ffi_api.BlockBuilderEndBlock(self._ib)
        if len(block.bindings) > 0:
            self._ib._blocks.append(block)


class DataflowScope(object):
    """Auxiliary scope for Dataflow block"""

    def __init__(self, irbuilder):
        self._ib = irbuilder

    def __enter__(self):
        block = _ffi_api.BlockBuilderEndBlock(self._ib)
        if len(block.bindings) > 0:
            self._ib._blocks.append(block)
        _ffi_api.BlockBuilderBeginDataflowBlock(self._ib)

    def __exit__(self, ptype, value, trace):
        block = _ffi_api.BlockBuilderEndBlock(self._ib)
        if len(block.bindings) > 0:
            self._ib._blocks.append(block)
        _ffi_api.BlockBuilderBeginBindingBlock(self._ib)


@tvm._ffi.register_object("relax.BlockBuilder")
class BlockBuilder(Object):
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
        with ib.function([x, y], "func"):
            with ib.dataflow() as df:
                lv0 = ib.emit(rx.add(x, y))
                lv1 = ib.emit(rx.multiply(lv0, y))
                gv0 = ib.emit_output(lv1)
            ib.emit_func_output(gv0)
        func = ib.get()
    """

    def __init__(self):
        self._blocks = []
        self.__init_handle_by_constructor__(_ffi_api.BlockBuilderCreate)

    def _begin_dataflow_block(self) -> None:
        _ffi_api.BlockBuilderBeginDataflowBlock(self)
    
    def _begin_binding_block(self) -> None:
        _ffi_api.BlockBuilderBeginBindingBlock(self)
    
    def _end_block(self) -> BindingBlock:
        return _ffi_api.BlockBuilderEndBlock(self)

    def function(self,
                 params: Optional[Union[Var, Tuple, List[Var]]] = None,
                 name: Optional[str] = "") -> FunctionScope:
        """Annotate a Relax function.

        Parameters
        ----------
        params : tvm.relax.Var | Tuple | List[tvm.relax.Var], optional
            The parameters of the function.

        name : str, optional
            The name of the function. If provided, the function is global, otherwise local.

        Returns
        -------
        ret: FunctionScope
            A FunctionScope for building a Relax function node.
        """
        if not params:
            params = []
        if not isinstance(params, (list, tuple)):
            params = [params]

        self._func_params = params
        self._func_name = name
        return FunctionScope(self)

    def dataflow(self) -> DataflowScope:
        """Annotate a Relax dataflow block.

        Returns
        -------
        ret: DataflowScope
            A DataflowScope for building a Relax dataflow block.
        """
        return DataflowScope(self)

    def emit(self, call: relay.Call) -> Var:
        """Emit a call node.
        This infers the shape and type of the CallNode, create a variable,
        and bind the CallNode to the variable.

        Parameters
        ----------
        call : tvm.relay.Call
            The call node to be emitted.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets binded to the call code.
        """
        return _ffi_api.BlockBuilderEmit(self, call)

    def match_shape(self, value: Expr, pattern: List[PrimExpr]) -> Var:
        """Emit a MatchShape.

        Parameters
        ----------
        value : tvm.relay.Expr
            The value of the MatchShape to be emitted.

        pattern : List[PrimExpr]
            The pattern of the MatchShape to be emitted.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets binded to the call code.
        """
        return _ffi_api.BlockBuilderEmitMatchShape(self, value, pattern)

    def emit_output(self, output: Union[Expr, Tuple, List[Expr]]) -> None:
        """Emit output for the current dataflow block or function.

        Parameters
        ----------
        output : Expr | Tuple | List[Expr]
            The output of the current block/function.

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets binded to the output.
        """
        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        return _ffi_api.BlockBuilderEmitOutput(self, output)

    def emit_func_output(self, output: Union[Expr, Tuple, List[Expr]]) -> None:
        """Emit output for the function.

        Parameters
        ----------
        output : Expr | Tuple | List[Expr]
            The output of the current block/function.

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets binded to the output.
        """
        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        self._func_ret = output

    def normalize(self, expr: Expr) -> Expr:
        """Normalize an Expr to complete its shape and type.

        Parameters
        ----------
        expr : Expr
            The input expr.

        Returns
        -------
        ret : Expr
            The expr with normalized shape and type.
        """
        return _ffi_api.BlockBuilderNormalize(self, expr)

    def get(self) -> Function:
        """Return the function being built.

        Returns
        -------
        ret : tvm.relax.Function
            A Relax function node being built.
        """
        seqe = rx.SeqExpr(self._blocks, self._func_ret)
        func = rx.Function(
            self._func_params, seqe, rx.DynTensorType(-1, "float32"), rx.GlobalVar(self._func_name)
        )
        return func
