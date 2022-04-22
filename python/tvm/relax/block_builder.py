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
"""Developer API of constructing Relax AST."""
import typing

from typing import List, Optional, Union, Any, Callable
from tvm.runtime import Object
from tvm import relax as rx, tir
import tvm
from .expr import (
    Expr,
    te_tensor,
    Var,
    ShapeExpr,
    GlobalVar,
    PrimExpr,
    BindingBlock,
    Tuple,
    BaseFunc,
)
from .op.base import call_tir
from . import _ffi_api


class FunctionScope(object):
    """Auxiliary scope for function"""

    def __init__(self, block_builder, name, params):
        self._bb = block_builder
        self._name = name
        self._params = params

    def __enter__(self):
        self._bb._enter_function_scope(self._name, self._params)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # __exit__ should properly handle the case where the with block exits with an exception
        # when handling error case in exit, always check if there is already an exception
        # been thrown in the with block
        self._bb._exit_function_scope(exc_type, exc_val, exc_tb)


class DataflowScope(object):
    """Auxiliary scope for Dataflow block"""

    def __init__(self, block_builder):
        self._bb = block_builder

    def __enter__(self):
        block = self._bb._end_block()
        if len(block.bindings) > 0:
            self._bb._blocks.append(block)
        self._bb._begin_dataflow_block()

    def __exit__(self, ptype, value, trace):
        block = self._bb._end_block()
        if len(block.bindings) > 0:
            self._bb._blocks.append(block)
        self._bb._begin_binding_block()


@tvm._ffi.register_object("relax.BlockBuilder")
class BlockBuilder(Object):
    """A builder to build Relax IR for testing and dev.

    Examples
    --------
    .. code-block:: python

        m = tir.Var("m", "int32")
        n = tir.Var("n", "int32")
        type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
        type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")
        x = rx.Var("x", [m, n], type_anno0)
        y = rx.Var("y", [n], type_anno1)
        bb = rx.BlockBuilder()
        with bb.function([x, y], "func"):
            with bb.dataflow() as df:
                lv0 = bb.emit(rx.add(x, y))
                lv1 = bb.emit(rx.multiply(lv0, y))
                gv0 = bb.emit_output(lv1)
            bb.emit_func_output(gv0)
        mod = bb.get()

    BlockBuilder can also be used to contruct neural networks with nn.Module API

    .. code-block:: python

        from tvm.relax.testing import nn

        n = tir.Var("n", "int64")
        input_size = 784
        hidden_sizes = [128, 32]
        output_size = 10
        bb = rx.BlockBuilder()

        with bb.function("main"):
            model = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], output_size),
                nn.LogSoftmax(),
            )
            data = nn.Placeholder((n, input_size), name="data")
            output = model(data)
            params = [data] + model.parameters()
            builder.emit_func_output(output, params=params)
        mod = bb.get()
    """

    _current = None

    @staticmethod
    def current():
        """Returns the current BlockBuilder."""
        return BlockBuilder._current

    def __init__(self):
        self._blocks = []
        # a boolean flag that tracks if emit_func_output has been called
        self._is_emit_func_output_called = False
        self.__init_handle_by_constructor__(_ffi_api.BlockBuilderCreate)

    def _begin_dataflow_block(self) -> None:
        _ffi_api.BlockBuilderBeginDataflowBlock(self)

    def _begin_binding_block(self) -> None:
        _ffi_api.BlockBuilderBeginBindingBlock(self)

    def _end_block(self) -> BindingBlock:
        return _ffi_api.BlockBuilderEndBlock(self)

    def _enter_function_scope(self, name, params):
        if BlockBuilder.current() is not None:
            raise RuntimeError("BlockBuilder does not allow nested functions.")
        BlockBuilder._current = self
        self._func_name = name
        self._func_params = params
        self._begin_binding_block()

    def _exit_function_scope(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if not self._is_emit_func_output_called:
                raise RuntimeError("emit_func_output must be called in a relax function.")

        self._blocks = []
        self._is_emit_func_output_called = False
        BlockBuilder._current = None

    def _convert_te_arg(self, te_args: Any) -> typing.Tuple[Any, List[tvm.te.Tensor]]:
        """Helper function to convert Relax expressions to te tensor.
        In the common case, the type of te_args is a Relax expression and is converted
        into a te tensor.
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
            A tuple of the converted te_args, and a list of te tensors for each converted
            Relax expression
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
                    assert isinstance(
                        key, str
                    ), "emit_te only supports dict with string as the key currently"
                return {k: _convert_te_arg_helper(arg[k]) for k in arg}
            elif (
                isinstance(arg, (int, float, str, tir.IntImm, tvm.ir.Type, tvm.ir.Attrs))
                or arg is None
            ):
                return arg
            raise TypeError("not supported type in emit_te: {}".format(type(arg)))

        new_arg = _convert_te_arg_helper(te_args)
        return new_arg, te_args_list

    def _get_unbound_tir_vars(self, args: List[tvm.te.Tensor]) -> List[tvm.tir.Var]:
        """get unbound TIR vars (i.e TIR vars used in the shape but is not
        itself a dimension of a shape)"""
        bound_vars = set()
        used_vars = set()

        def _populate_used_vars(expr):
            if isinstance(expr, tvm.tir.Var):
                used_vars.add(expr)

        for x in args:
            for s in x.shape:
                tvm.tir.stmt_functor.post_order_visit(s, _populate_used_vars)
                if isinstance(s, tir.Var):
                    bound_vars.add(s)

        diff = used_vars - bound_vars
        return list(diff)

    def function(
        self, name: str, params: Optional[Union[Var, Tuple, List[Var]]] = None
    ) -> FunctionScope:
        """Annotate a Relax function.

        Parameters
        ----------
        name : str, optional
            The name of the function

        params : tvm.relax.Var | Tuple | List[tvm.relax.Var], optional
            The parameters of the function.
            If params is None, it means deferring initialization of function parameters
            until emit_func_output.

        Returns
        -------
        ret: FunctionScope
            A FunctionScope for building a Relax function node.
        """
        if not params:
            params = None
        elif isinstance(params, rx.Var):
            params = [params]
        elif isinstance(params, (list, tuple)):
            for param in params:
                if not isinstance(param, rx.Var):
                    raise TypeError(
                        "each element of function parameters must be of type tvm.relax.Var,\
                                    but got: {}".format(
                            type(param)
                        )
                    )

        return FunctionScope(self, name, params)

    def dataflow(self) -> DataflowScope:
        """Annotate a Relax dataflow block.

        Returns
        -------
        ret: DataflowScope
            A DataflowScope for building a Relax dataflow block.
        """
        return DataflowScope(self)

    def emit(self, expr: Expr) -> Var:
        """Emit an expr.
        This infers the shape and type of the expr, create a variable,
        and bind the expr to the variable.

        Parameters
        ----------
        expr : tvm.relax.Expr
            The Expr to be emitted.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets bound to the input expr.
        """
        return _ffi_api.BlockBuilderEmit(self, expr)

    def call_te(self, func: Callable, *args: Any, **kwargs: Any) -> Expr:
        """Generate a call node according to the te function.
        This function converts arguments from relax expression to te tensor,
        The callback func should return a te tensor or a list of te tensors.
        Please see detailed example in emit_te

        Parameters
        ----------
        func : Callable
            A function that returns a te tensor or a list of te tensors.

        args : Any, optional
            arguments passed to the function.

        kwargs : Any, optional
            The keyword arguments passed to the function.
            Note that the key "primfunc_name_hint" is reserved for passing name hint
            to the PrimFunc that gets generated.

        Returns
        -------
        ret : tvm.relax.Call
            A newly created call node
        """

        primfunc_name_hint = kwargs.pop("primfunc_name_hint", None)
        new_args, te_arg_list = self._convert_te_arg(args)
        new_kwargs, te_kwarg_list = self._convert_te_arg(kwargs)

        te_args = te_arg_list + te_kwarg_list

        te_out = func(*new_args, **new_kwargs)
        assert isinstance(te_out, tvm.te.tensor.Tensor) or (
            isinstance(te_out, (tuple, list, tvm.ir.Array))
            and all(isinstance(t, tvm.te.tensor.Tensor) for t in te_out)
        ), "only support te.tensor or tuple/list/Array of te.tensor as function output"

        if isinstance(te_out, (tuple, list, tvm.ir.Array)) and len(te_out) == 1:
            te_out = te_out[0]

        outs = [te_out] if isinstance(te_out, tvm.te.tensor.Tensor) else list(te_out)
        unbound_tir_vars = self._get_unbound_tir_vars(te_args + outs)

        inputs = [*te_args] + outs
        tir_func = tvm.te.create_prim_func(inputs, unbound_tir_vars)

        if primfunc_name_hint:
            gvar = self.add_func(tir_func, primfunc_name_hint)
        else:
            gvar = self.add_func(tir_func, func.__name__)

        call_args = [x.op.value for x in te_args]

        output_shape = (
            outs[0].shape
            if isinstance(te_out, tvm.te.tensor.Tensor)
            else Tuple([ShapeExpr(x.shape) for x in outs])
        )

        output_dtype = (
            te_out.dtype if isinstance(te_out, tvm.te.tensor.Tensor) else [x.dtype for x in outs]
        )

        # add arguments for extra parameters from unbound var
        if len(unbound_tir_vars) > 0:
            call = call_tir(
                gvar, call_args, output_shape, output_dtype, tir_vars=ShapeExpr(unbound_tir_vars)
            )
        else:
            call = call_tir(gvar, call_args, output_shape, output_dtype)
        return call

    def emit_te(self, func: Callable, *args: Any, **kwargs: Any) -> Var:
        """Emit a call node according to the te function.
        This function converts arguments from relax expression to te tensor,
        The callback func should return a te tensor or a list of te tensors.

        Parameters
        ----------
        func : Callable
            A function that returns a te tensor or a list of te tensors.

        args : Any, optional
            arguments passed to the function.

        kwargs : Any, optional
            The keyword arguments passed to the function.
            Note that the key "primfunc_name_hint" is reserved for passing name hint
            to the PrimFunc that gets generated.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets bound to the call code.

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
                def te_func(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle,
                            var_compute: T.handle) -> None:
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
                def rx_func(x: Tensor((n, m), "float32"), y: Tensor((n, m), "float32")) -> Tensor:
                    # block 0
                    gv = relax.call_tir("te_func", (x, y), (128, 128), dtype="float32")
                    return gv

        Example
        -------

        .. code-block:: python

            bb = relax.BlockBuilder()
            n = tir.Var("n", "int64")
            type_anno = relax.DynTensorType(1, "float32")
            x = relax.Var("x", [n], type_anno)
            y = relax.Var("y", [n + 1], type_anno)

            def te_func(A):
                C = te.compute((n + 1), lambda i: A[i])
                return C

            with bb.function("rx_func", [x, y]):
                x1 = bb.emit_te(te_func, y)
                bb.emit_func_output(x1)

        will result in TVMScript

        .. code-block:: python

            @tvm.script.ir_module
            class Module:
                @T.prim_func
                def te_func(var_rxplaceholder: T.handle, var_compute: T.handle, n: T.int64) -> None:
                    # function attr dict
                    T.func_attr({"global_symbol": "te_func"})
                    rxplaceholder = T.match_buffer(var_rxplaceholder, [n + T.int64(1)],
                                                   dtype="float32")
                    compute = T.match_buffer(var_compute, [n + T.int64(1)], dtype="float32")
                    # body
                    # with T.block("root")
                    for i0 in T.serial(0, n + T.int64(1)):
                        with T.block("compute"):
                            i = T.axis.spatial(n + T.int64(1), i0)
                            T.reads([rxplaceholder[i]])
                            T.writes([compute[i]])
                            compute[i] = rxplaceholder[i]

                @R.function
                def rx_func(x: Tensor((n,), "float32"), y: Tensor(((n + 1),), "float32"))
                    -> Tensor(None, "float32", ndim=-1):
                    # block 0
                    gv = relax.call_tir(te_func, (y,), ((n + 1),), (n,), dtype="float32")
                    return gv
        """
        return self.emit(self.call_te(func, *args, **kwargs))

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
            A newly created variable that gets bound to the call code.
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
            The return variable which gets bound to the output.
        """
        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        return _ffi_api.BlockBuilderEmitOutput(self, output)

    def emit_func_output(
        self,
        output: Union[Expr, Tuple, List[Expr]],
        params: Optional[Union[Var, Tuple, List[Var]]] = None,
    ) -> None:
        """Emit output for the function.

        Parameters
        ----------
        output : Expr | Tuple | List[Expr]
            The output of the current block/function.

        params : tvm.relax.Var | Tuple | List[tvm.relax.Var], optional
            The parameters of the function to be built.
            If params is None, it means the params have been initialized in the function with scope.

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets bound to the output.
        """
        if self._is_emit_func_output_called:
            raise RuntimeError("emit_func_output must be called exactly once in a relax function.")
        self._is_emit_func_output_called = True

        if self._func_params is not None and params is not None:
            raise RuntimeError(
                "function parameters have been initialized in the function with scope."
            )

        if self._func_params is None and params is None:
            raise RuntimeError("Relax function must have parameter.")

        if self._func_params is None:
            self._func_params = params

        if BlockBuilder.current() is not self:
            raise RuntimeError("BlockBuilder._current must be self.")

        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        self._func_ret = output

        block = self._end_block()
        if len(block.bindings) > 0:
            self._blocks.append(block)
        seqe = rx.SeqExpr(self._blocks, self._func_ret)
        # TODO: rx function can return Tuple as well, need to infer ret type
        func = rx.Function(
            self._func_params, seqe, rx.DynTensorType(-1), rx.GlobalVar(self._func_name)
        )
        self.add_func(func, self._func_name)

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
        return _ffi_api.BlockBuilderGetContextIRModule(self)

    def get_unique_name(self, name_prefix: str) -> str:
        """Generate a unique name with a specified prefix.

        Parameters
        ----------
        name_hint : str
            The name prefix.

        Returns
        -------
        ret : str
            The generated name.
        """
        return _ffi_api.BlockBuilderGetUniqueName(self, name_prefix)

    def add_func(self, func: BaseFunc, func_name: str) -> GlobalVar:
        """Add a Relax function or a TIR PrimFunc to the IRModule being built.

        Parameters
        ----------
        func : BaseFunc
            The function to be added.

        func_name : str
            The name of the function to be added.

        Returns
        -------
        gvar : GlobalVar
            The global var bound to the added function.
        """
        return _ffi_api.BlockBuilderAddFuncToContext(self, func, func_name)
