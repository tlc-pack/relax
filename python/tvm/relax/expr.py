# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-import, super-init-not-called
# pylint: disable=redefined-builtin
"""The expression nodes of Relax."""
from typing import Any, List, Optional, Union
import typing
import numpy as _np  # type: ignore

import tvm
import tvm._ffi
from tvm.runtime import ndarray as _nd
import tvm.relax

from tvm._ffi import base as _base
from .. import relay
from ..ir import BaseFunc, Node, SourceName, Span
from ..relay import Id
from ..runtime import String
from ..tir import PrimExpr
from . import _ffi_api, ty

# It is a workaround for mypy: https://github.com/python/mypy/issues/7866#issuecomment-549454370
# This feature is not supported until python 3.10:
# https://docs.python.org/3.10/whatsnew/3.10.html#pep-613-typealias
Expr = Union[relay.Expr]
Type = Union[relay.Type]
GlobalVar = Union[relay.GlobalVar]

# NOTE: place base struct info in expr to avoid cyclic dep
# from expr to struct info.
class StructInfo(Node):
    """The base class of all StructInfo.

    StructInfo contains both the static type
    and runtime structural information.
    """

    def __eq__(self, other):
        """Compare two struct info for structural equivalence."""
        return tvm.ir.structural_equal(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Overload with structural equality."""
        return super().__eq__(other)

    def is_base_of(self, derived: "StructInfo") -> bool:
        """Check if self is base of another derived struct info.

        Parameters
        ----------
        derived : StructInfo
            The derived struct info to be checked.

        Returns
        -------
        result : bool
            The check result.
        """
        return _ffi_api.StructInfoIsBaseOf(self, derived)  # type: ignore


@tvm._ffi.register_object("relax.expr.Call")
class Call(Expr):
    """Function call node in Relax.

    Call node corresponds the operator application node
    in computational graph terminology.

    Parameters
    ----------
    op: tvm.ir.Op or any tvm.relax.Expr with function type.
        The operation to be called.

    args: Union[List[Expr], typing.Tuple[Expr, ...]]
        The arguments to the call.

    attrs: Optional[tvm.ir.Attrs]
        Attributes to the call, can be None

    type_args: Optional[Union[List[Type], typing.Tuple[Type, ...]]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.

    span: Optional[Span]
        Span that points to original source code
    """

    def __init__(
        self,
        op: Union[Expr, tvm.ir.Op],
        args: Union[List[Expr], typing.Tuple[Expr, ...]],
        attrs: Optional[tvm.ir.Attrs] = None,
        type_args: Optional[Union[List[Type], typing.Tuple[Type, ...]]] = None,
        span: Optional[Span] = None,
    ):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(
            _ffi_api.Call, op, args, attrs, type_args, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.If")
class If(Expr):
    """A conditional expression in Relax.

    Parameters
    ----------
    cond: Expr
        The condition.

    true_branch: Expr
        The expression evaluated when condition is true.

    false_branch: Expr
        The expression evaluated when condition is false.
    """

    def __init__(self, cond: Expr, true_branch: Expr, false_branch: Expr, span: Span = None):
        self.__init_handle_by_constructor__(
            _ffi_api.If, cond, true_branch, false_branch, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.Tuple")
class Tuple(Expr):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : Union[List[Expr], typing.Tuple[Expr, ...]]
        The fields in the tuple.

    span: Optional[Span]
        Span that points to original source code
    """

    def __init__(self, fields: Union[List[Expr], typing.Tuple[Expr, ...]], span: Span = None):
        self.__init_handle_by_constructor__(_ffi_api.Tuple, fields, span)  # type: ignore

    def __getitem__(self, index: int) -> Expr:
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self) -> int:
        return len(self.fields)


@tvm._ffi.register_object("relax.expr.TupleGetItem")
class TupleGetItem(Expr):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: Expr
        The input tuple expression.

    index: int
        The index.
    """

    def __init__(self, tuple_value: Expr, index: int):
        self.__init_handle_by_constructor__(
            _ffi_api.TupleGetItem, tuple_value, index  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.ShapeExpr")
class ShapeExpr(Expr):
    """A shape expression which allows users to construct a shape containing PrimExpr."""

    values: List[PrimExpr]

    def __init__(
        self,
        values: Union[List[PrimExpr], typing.Tuple[PrimExpr, ...], tvm.ir.Array],
        span: Span = None,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeExpr, values, span)  # type: ignore

    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            raise IndexError("Tuple index out of range")
        return self.values[index]

    def __len__(self):
        return len(self.values)


def make_shape(shape: Union[List[Any], typing.Tuple[Any, ...]]) -> ShapeExpr:
    if isinstance(shape, (list, tuple)):
        return ShapeExpr(shape)
    raise ValueError("Wrong type")


@tvm._ffi.register_object("relax.expr.Constant")
class Constant(Expr):
    def __init__(self, data: tvm.nd.NDArray, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Constant, data, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.Var")
class Var(Expr):
    """The variable class for all Relax bindings."""

    vid: Id
    struct_info: Optional[StructInfo]

    def __init__(
        self,
        name_hint: str,
        struct_info: Optional[StructInfo] = None,
        span: Span = None,
    ) -> None:
        if struct_info is not None and not isinstance(struct_info, StructInfo):
            raise TypeError(
                "struct_info needs to be an instance of StructInfo. "
                "If you attempt to pass in shape, "
                "use relax.TensorStructInfo(shape, dtype)."
            )
        self.__init_handle_by_constructor__(
            _ffi_api.Var if isinstance(name_hint, str) else _ffi_api.VarFromId,  # type: ignore
            name_hint,
            struct_info,
            span,
        )

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name

    def __call__(self, *args: Any, attrs=None) -> Call:
        if self._checked_type_ and isinstance(self._checked_type_, ty.FuncType):
            return Call(self, args, attrs=attrs)
        else:
            raise TypeError(
                f"Only vars with function type can be called, but got type: {self._checked_type_}"
            )

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("TupleGetItem only supports integer index")
        var_type = self._checked_type_
        if var_type and isinstance(var_type, ty.TupleType):
            return TupleGetItem(self, key)
        else:
            raise TypeError(
                f"Only vars with TupleType is subscriptable, but got type: {self._checked_type_}"
            )


@tvm._ffi.register_object("relax.expr.DataflowVar")
class DataflowVar(Var):
    """A sub-type of the variable node used to mark dataflow variables from
    normal visible "function local" bindings."""

    vid: Id
    struct_info: Optional[StructInfo]

    def __init__(
        self,
        name_hint: Union[str, Id],
        struct_info: Optional[StructInfo] = None,
        span: Span = None,
    ) -> None:
        if struct_info is not None and not isinstance(struct_info, StructInfo):
            raise TypeError(
                "struct_info needs to be an instance of StructInfo. "
                "If you attempt to pass in shape, "
                "use relax.TensorStructInfo(shape, dtype)."
            )

        self.__init_handle_by_constructor__(
            _ffi_api.DataflowVar  # type: ignore
            if isinstance(name_hint, str)
            else _ffi_api.DataflowVarFromId,  # type: ignore
            name_hint,
            struct_info,
            span,
        )


@tvm._ffi.register_object("relax.expr.Binding")
class Binding(Node):
    """The base class of a binding in Relax."""

    ...


@tvm._ffi.register_object("relax.expr.MatchCast")
class MatchCast(Binding):
    """Runtime-match the value to the struct info.

    This operation does runtime check, populates the un-defined symbolic shape vars
    and vars in struct_info in the first occurrence, and insert equality assertions in
    other cases.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    struct_info: tvm.relax.StructInfo
        The struct info to match cast to.
    """

    var: Var
    struct_info: "tvm.relax.StructInfo"
    value: Expr

    def __init__(
        self, var: Var, value: Expr, struct_info: "tvm.relax.StructInfo", span: Span = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchCast, var, value, struct_info, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.VarBinding")
class VarBinding(Binding):
    """Variable binding, bind he variable of the lhs with the rhs."""

    var: Var
    value: Expr

    def __init__(self, var: Var, value: Expr, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.VarBinding, var, value, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.BindingBlock")
class BindingBlock(Node):
    """base class of binding block, bindings inside can be impure
    (with side effect or control flow)"""

    bindings: List[Binding]

    def __init__(self, bindings: List[Binding], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BindingBlock, bindings, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(BindingBlock):
    """dataflow block, bindings inside are pure (no side effect and no control flow)"""

    def __init__(self, bindings: List[Binding], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock, bindings, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(Expr):
    """A sequence of binding blocks followed by an expression."""

    blocks: List[BindingBlock]
    body: Expr

    def __init__(self, blocks: List[BindingBlock], body: Expr, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.Function")
class Function(BaseFunc):
    """A Relax function."""

    params: List[Var]
    body: Expr
    ret_struct_info: StructInfo
    attrs: Optional[tvm.ir.DictAttrs]

    def __init__(
        self,
        params: List[Var],
        body: Expr,
        ret_struct_info: Optional[StructInfo] = None,
        attrs: Optional[tvm.ir.DictAttrs] = None,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Function, params, body, ret_struct_info, attrs, span  # type: ignore
        )

    @staticmethod
    def create_empty(
        params: List[Var],
        ret_struct_info: StructInfo,
        attrs: Optional[tvm.ir.DictAttrs] = None,
        span: Optional[Span] = None,
    ):
        """Construct a relax.Function but without body"""
        return _ffi_api.FunctionCreateEmpty(params, ret_struct_info, attrs, span)  # type: ignore

    def __call__(self, *args):
        """Invoke the global function.

        Parameters
        ----------
        args: List[relax.Expr]
            Arguments.
        """
        return Call(self, args, None, None)

    def script(self, show_meta: bool = False) -> str:
        """Print relax.Function into TVMScript

        Parameters
        ----------
        show_meta : bool
            Whether to show meta information

        Returns
        -------
        script : str
            The TVM Script of the relax.Function
        """
        return tvm._ffi.get_global_func("script.AsRelaxScript")(self, show_meta)  # type: ignore

    def show(self, style: str = "light") -> None:
        """
        A sugar for print highlighted TVM script.

        Parameters
        ----------
        style : str, optional
            Pygments styles extended by "light" (default) and "dark", by default "light"
        """
        from tvm.script.highlight import cprint  # pylint: disable=import-outside-toplevel

        # Use deferred import to avoid circular import while keeping cprint under tvm/script
        cprint(self, style=style)


@tvm._ffi.register_object("relax.expr.ExternFunc")
class ExternFunc(BaseFunc):
    """extern function, which can represent a TIR PrimFunc or a PackedFunc."""

    global_symbol: String

    def __init__(self, global_symbol: String, span: Span = None) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExternFunc, global_symbol, span  # type: ignore
        )


def extern(name: str, span: Span = None):
    """Create extern function."""
    return ExternFunc(name, span)


def const(
    value: Union[bool, int, float, _np.ndarray, tvm.nd.NDArray], dtype: Optional[str] = None
) -> Constant:
    """Create a constant value.

    Parameters
    ----------
    value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
        The constant value.

    dtype: Optional[str]
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    if isinstance(value, (_base.numeric_types, (bool, list))):
        value = _np.array(value, dtype=dtype)

    if not dtype:
        # when dtype is None: int maps to "int32", float maps to "float32"
        dtype = {  # type: ignore
            _np.dtype("int64"): _np.int32,  # type: ignore
            _np.dtype("float64"): _np.float32,  # type: ignore
        }.get(
            value.dtype, None  # type: ignore
        )

    if isinstance(value, (_np.ndarray, _np.generic)):
        if dtype is not None:
            value = value.astype(dtype)
        value = _nd.array(value)

    if not isinstance(value, _nd.NDArray):
        raise ValueError("value has to be scalar or NDArray")

    return Constant(value)


def te_tensor(value: Expr, name: str = "rxplaceholder"):
    """Create te tensor from relax expression."""
    return _ffi_api.TETensor(value, name)  # type: ignore


def get_shape_of(expr: Expr) -> Expr:
    """Get shape of expr.

    Parameters
    ----------
    expr: Expr
        The input expr.

    Returns
    -------
    shape: Expr
        The shape expression

    Note
    ----
    This function requires expr to be normalized.
    The function will report an error if expr's StructInfo is not TensorStructInfo.
    It will try to return symbolic function when possible. If the tensor do not
    have a compile-time symbolic shape, the function will then choose to return
    `Call(relax.op.shape_of, [expr])`.
    """
    return _ffi_api.GetShapeOf(expr)  # type: ignore


def _update_struct_info(expr: Expr, struct_info: Optional[StructInfo]) -> None:
    _ffi_api.UpdateStructInfo(expr, struct_info)  # type: ignore
