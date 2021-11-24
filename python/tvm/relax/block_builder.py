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
import typing
from typing import List, Optional, Union, Dict, Any, Callable
from tvm.relay.expr import Tuple
from tvm.runtime import Object
from tvm import relax as rx
from tvm import tir
from .expr import *
from .op.base import call_dps
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
        seqe = rx.SeqExpr(self._ib._blocks, self._ib._func_ret)
        func = rx.Function(
            self._ib._func_params, seqe, rx.DynTensorType(-1, "float32"), rx.GlobalVar(self._ib._func_name)
        )
        gvar = rx.GlobalVar(self._ib._func_name)
        self._ib._context_mod[gvar] = func
        return func


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
        ib = rx.BlockBuilder()
        with ib.function([x, y], "func"):
            with ib.dataflow() as df:
                lv0 = ib.emit(rx.add(x, y))
                lv1 = ib.emit(rx.multiply(lv0, y))
                gv0 = ib.emit_output(lv1)
            ib.emit_func_output(gv0)
        mod = ib.get()
    """

    def __init__(self):
        self._blocks = []
        self._context_mod = tvm.IRModule()
        self.__init_handle_by_constructor__(_ffi_api.BlockBuilderCreate)

    def _begin_dataflow_block(self) -> None:
        _ffi_api.BlockBuilderBeginDataflowBlock(self)
    
    def _begin_binding_block(self) -> None:
        _ffi_api.BlockBuilderBeginBindingBlock(self)

    def _end_block(self) -> BindingBlock:
        return _ffi_api.BlockBuilderEndBlock(self)

    def _convert_te_arg(self,
        te_args: Any
    ) -> typing.Tuple[Any, List[tvm.te.Tensor]]:
        """Helper function to convert Relax expressions to te tensor.
        In the common case, the type of te_args is a Relax expression and is converted into a te tensor.
        If te_args is a nested or recursive datatype (i.e list, dict, tvm.ir.Map, tvm.ir.Array), 
        we recursive and convert any value of type Relax expression into a te tensor. 
        Common values of type int, float, and str are preserved.

        Parameters
        ----------
        te_args : Any
            Argument to convert to te

        Returns
        -------
        ret : (Any, [tvm.te.Tensor])
            A tuple of the converted te_args, and a list of te tensors for each converted Relax expression
        """
        te_args_list = []

        def _convert_te_arg_helper(arg):
            if isinstance(arg, Expr):
                arg = te_tensor(arg)
                te_args_list.append(arg)
                return arg
            elif isinstance(arg, (list, tvm.ir.Array)):
                return [_convert_te_arg_helper(x) for x in arg]
            elif isinstance(arg, tuple):
                return tuple([_convert_te_arg_helper(x) for x in arg])
            elif isinstance(arg, (dict, tvm.ir.Map)):
                for key in arg:
                    assert isinstance(key, str), "emit_te only supports dict with string as the key currently"
                return {k: _convert_te_arg_helper(arg[k]) for k in arg}
            elif isinstance(arg, (int, float, str)):
                return arg
            else:
                raise TypeError("not supported type in emit_te: {}".format(type(arg)))

        new_arg = _convert_te_arg_helper(te_args)
        return new_arg, te_args_list
    
    def _check_te_args(self, args: List[tvm.te.Tensor]):
        """check te arguments."""
        #TODO(hypercubestart, ziheng) support full dynamic shape in the future
        for x in args:
            for s in x.shape:
                if not isinstance(s, (tir.Var, tir.IntImm)):
                    raise ValueError("emit_te not support symbolic shape"
                        "contains expression now: {}".format(x.shape))

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
        call : tvm.relax.Call
            The call node to be emitted.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets binded to the call code.
        """
        return _ffi_api.BlockBuilderEmit(self, call)

    def emit_te(self, func: Callable, *args: Any, **kwargs: Any) -> Var:
        """Emit a call node according to the te function.
        This function converts arguments from relax expression to te tensor,
        The callback func should return a te tensor.

        Parameters
        ----------
        func : Callable
            A function that return a te tensor.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets binded to the call code.

        Example
        -------

        .. code-block:: python

            bb = rx.BlockBuilder()
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            type_anno = rx.DynTensorType(2, "float32")
            x = rx.Var("x", [n, m], type_anno)
            y = rx.Var("y", [n, m], type_anno)
            
            def te_func(args, args_dict, msg):
                A = args[0]
                B = args_dict["B"]
                return te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
            
            with bb.function([x, y], "rx_func"):
                out = bb.emit_te(te_func, [x], {"B": y}, msg="hello")
                bb.emit_func_output(out)

        will result in TVMScript

        .. code-block:: python

            @tvm.script.ir_module
            class Module:
                @T.prim_func
                def te_func(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_compute: T.handle) -> None:
                    # function attr dict
                    T.func_attr({"global_symbol": "te_func"})
                    m = T.var("int64")
                    n = T.var("int64")
                    rxplaceholder = T.match_buffer(var_rxplaceholder, [n, m], dtype="float32")
                    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [n, m], dtype="float32")
                    compute = T.match_buffer(var_compute, [128, 128], dtype="float32")
                    # body
                    # with T.block("root")
                    for i0, i1 in T.grid(128, 128):
                        with T.block("compute"):
                            i, j = T.axis.remap("SS", [i0, i1])
                            T.reads([rxplaceholder[i, j], rxplaceholder_1[i, j]])
                            T.writes([compute[i, j]])
                            compute[i, j] = rxplaceholder[i, j] + rxplaceholder_1[i, j]

                @R.function
                def rx_func(x: Tensor[(n, m), "float32"], y: Tensor[(n, m), "float32"]) -> Tensor:
                    # block 0
                    gv = relax.call_dps((128, 128), "te_func", (x, y))
                    return gv
        """
        new_args, te_arg_list = self._convert_te_arg(args)
        new_kwargs, te_kwarg_list = self._convert_te_arg(kwargs)

        te_args = te_arg_list + te_kwarg_list
        self._check_te_args(te_args)

        # TODO(hypercubestart, ziheng) handle multiple output case
        te_out = func(*new_args, **new_kwargs)
        assert isinstance(te_out, tvm.te.tensor.Tensor), "only support te tensor as function output"

        inputs = [*te_args, te_out]
        tir_func = tvm.te.create_prim_func(inputs)
        func_name = _ffi_api.BlockBuilderGetUniqueName(self, func.__name__)
        tir_func = tir_func.with_attr("global_symbol", func_name)
        gvar = GlobalVar(func_name)
        self._context_mod[gvar] = tir_func
        call = call_dps(inputs[-1].shape, gvar, [x.op.value for x in inputs[:-1]])
        return _ffi_api.BlockBuilderEmit(self, call)


    def match_shape(self, value: Expr, pattern: List[PrimExpr]) -> Var:
        """Emit a MatchShape.

        Parameters
        ----------
        value : tvm.relax.Expr
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

    def get(self) -> tvm.IRModule:
        """Return the IRModule being built.

        Returns
        -------
        ret : tvm.IRModule
            An IRModule with Relax and TIR functions being built.
        """
        return self._context_mod
