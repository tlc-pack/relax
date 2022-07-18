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

"""The Relax Pattern Language and tooling."""
# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=pointless-statement

from typing import List, Optional, Callable, Dict, Union, Tuple

import tvm
import tvm._ffi as tvm_ffi
from tvm.relax import DataflowBlock, Expr, Var
from tvm.relay.op import get

from ...ir import make_node
from ...runtime import Object
from ...ir.base import Node
from . import _ffi as ffi


def register_df_node(type_key=None):
    """Register a Relax node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm_ffi.register_object("relax.dataflow_pattern." + type_key.__name__)(type_key)
    return tvm_ffi.register_object(type_key)


class DFPattern(Node):
    """Base class of all Patterns."""

    def __call__(self, *args):
        args = list(args)
        if len(args) == 1 and args[0] is None:
            args = None
        return CallPattern(self, args)

    def __or__(self, other):
        return OrPattern(self, other)

    def __and__(self, other):
        return AndPattern(self, other)

    def __add__(self, other):
        return is_op("relax.add")(self, other)

    def __sub__(self, other):
        return is_op("relax.subtract")(self, other)

    def __mul__(self, other):
        return is_op("relax.multiply")(self, other)

    def __truediv__(self, other):
        return is_op("relax.divide")(self, other)

    def __invert__(self):
        return deny(self)

    def has_attr(self, attrs: Dict[str, Object]):
        """
        Add an attribute constraint to this pattern

        Parameters
        ----------
        attrs: Dict[str, Object]

        Returns
        -------
        result: tvm.relax.dataflow_pattern.DFPattern
            The resulting AttrPattern
        """
        attrs = make_node("DictAttrs", **attrs)
        return AttrPattern(self, attrs)

    def has_type(self, ttype: tvm.ir.type.Type):
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        ttype: tvm.ir.type.Type
            The type to match

        Returns
        -------
        result: tvm.relax.dataflow_pattern.DFPattern
            The resulting TypePattern
        """
        return has_type(ttype, self)

    def has_dtype(self, dtype: str):
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        dtype: str
            The dtype to match

        Returns
        -------
        result: tvm.relax.dataflow_pattern.DFPattern
            The resulting DataTypePattern
        """
        return has_dtype(dtype, self)

    def has_shape(self, *args):
        return has_shape(*args, pattern=self)

    def match(self, expr, var2val=None) -> bool:
        """
        Match the given expression or function against this pattern.

        Parameters
        ----------
        expr : tvm.relax.Expr
            The expression to match.
        var2val : Optional[Dict[tvm.relax.Var, tvm.relax.Expr]]
            A mapping from Var to Expr for autojump (only for match_expr).

        Returns
        -------
        result: bool
            Whether or not the expression matches the pattern
        """
        return match_expr(self, expr, var2val)

    def optional(self, option_constructor: Callable[["DFPattern"], "DFPattern"]):
        """
        Create a optional user of this pattern.

        Parameters
        ----------
        option_constructor: function
            A function that takes a single Pattern parameter and returns
            a constructed pattern matching the option

        Returns
        -------
        result: tvm.relax.dataflow_pattern.DFPattern
            The resulting Pattern
        """
        return self | option_constructor(self)

    def has_rt_dep_shape(self):
        return self & has_rt_dep_shape()

    def used_by(
        self, other: Union["DFPattern", "UsedBySeq", "OnlyUsedBySeq"], index=-1
    ) -> "UsedBySeq":
        return used_by(self, other, index)

    def __xor__(self, other) -> "UsedBySeq":
        return self.used_by(other, -1)

    def only_used_by(
        self, other: Union["DFPattern", "UsedBySeq", "OnlyUsedBySeq"], index=-1
    ) -> "OnlyUsedBySeq":
        return only_used_by(self, other, index)

    def __rshift__(self, other) -> "OnlyUsedBySeq":
        return self.only_used_by(other, -1)

    def dup(self) -> "DFPattern":
        return ffi.dup_pattern(self)

    def fork_to(self, *args) -> None:
        for v in args:
            self ^ v


@register_df_node
class RuntimeDepShapePattern(DFPattern):
    """A pattern matching a Relax RuntimeDepShape."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.RuntimeDepShapePattern)


@register_df_node
class ExprPattern(DFPattern):
    """A pattern which matches an expression.

    Parameters
    ----------
    expr : tvm.relax.Expr
        The expression to match.
    """

    def __init__(self, expr: Expr):
        self.__init_handle_by_constructor__(ffi.ExprPattern, expr)


@register_df_node
class VarPattern(DFPattern):
    """A pattern for Var.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any VarNode.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.VarPattern, name_hint)


@register_df_node
class DataflowVarPattern(DFPattern):
    """A pattern for DataflowVar.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any VarNode.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.DataflowVarPattern, name_hint)


@register_df_node
class GlobalVarPattern(DFPattern):
    """A pattern for GlobalVar.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any GlobalVarNode.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.GlobalVarPattern, name_hint)


@register_df_node
class ExternFuncPattern(DFPattern):
    """A external function pattern.

    Parameters
    ----------
    global_symbol: str
        The name of the function. Optional, if not provided,
        the pattern will match any ExternFuncNode.
    """

    def __init__(self, global_symbol: str = ""):
        self.__init_handle_by_constructor__(ffi.ExternFuncPattern, global_symbol)


@register_df_node
class ConstantPattern(DFPattern):
    """A pattern matching a Relax Constant."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.ConstantPattern)


@register_df_node
class CallPattern(DFPattern):
    """A pattern matching a function call node.

    Parameters
    ----------
    op: relax.dataflow_pattern.DFPattern
        The operation to be called.

    args: List[relax.dataflow_pattern.DFPattern]
        The arguments to the call or None to match any arguments.

    """

    def __init__(
        self,
        op: "DFPattern",
        args: List["DFPattern"],
    ):
        self.__init_handle_by_constructor__(ffi.CallPattern, op, args)


@register_df_node
class FunctionPattern(DFPattern):
    """A pattern matching a function node in Relax.

    Parameters
    ----------
    params: List[relax.dataflow_pattern.DFPattern]
        The parameters to the Function or None to match any parameters.

    body: relax.dataflow_pattern.DFPattern
        The body fo the Function

    """

    def __init__(
        self,
        params: List["DFPattern"],
        body: "DFPattern",
    ):
        self.__init_handle_by_constructor__(ffi.FunctionPattern, params, body)


@register_df_node
class TuplePattern(DFPattern):
    """A patern matching a Relax Tuple.

    Parameters
    ----------
    fields : Array[tvm.relax.dataflow_pattern.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields: tvm.ir.container.Array):
        self.__init_handle_by_constructor__(ffi.TuplePattern, fields)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("TuplePattern index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on TuplePattern")


@register_df_node
class UnorderedTuplePattern(DFPattern):
    """A patern matching a Relax Tuple unorderedly.

    Parameters
    ----------
    fields : Array[tvm.relax.dataflow_pattern.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields: tvm.ir.container.Array):
        self.__init_handle_by_constructor__(ffi.UnorderedTuplePattern, fields)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("UnorderedTuplePattern index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)


@register_df_node
class TupleGetItemPattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    tuple_value: tvm.relax.dataflow_pattern.DFPattern
        The input tuple expression.

    index: Optional[int]
        The index to match; Default (None) to match a TupleGetItem with any index.
    """

    def __init__(self, tuple_value: "DFPattern", index: Optional[int] = None):
        match_index = index if index is not None else -1
        self.__init_handle_by_constructor__(ffi.TupleGetItemPattern, tuple_value, match_index)


@register_df_node
class OrPattern(DFPattern):
    """Create a Pattern that can match one of two conditions

    Parameters
    ----------
    left: tvm.relax.dataflow_pattern.DFPattern
        One possible matching pattern.
    right: tvm.relax.dataflow_pattern.DFPattern
        One possible matching pattern.
    """

    def __init__(self, left: "DFPattern", right: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.OrPattern, left, right)


@register_df_node
class AndPattern(DFPattern):
    """Create a Pattern that must match two conditions

    Parameters
    ----------
    left: tvm.relax.dataflow_pattern.DFPattern
        One must-matching pattern.
    right: tvm.relax.dataflow_pattern.DFPattern
        One must-matching pattern.
    """

    def __init__(self, left: "DFPattern", right: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.AndPattern, left, right)


@register_df_node
class NotPattern(DFPattern):
    """Create a Pattern that matches the negation of a condition.

     Parameters
    ----------
    reject: tvm.relax.dataflow_pattern.DFPattern
        The pattern to deny.
    """

    def __init__(self, reject: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.NotPattern, reject)


@register_df_node
class WildcardPattern(DFPattern):
    """A pattern which matches anything."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.WildcardPattern)


@register_df_node
class TypePattern(DFPattern):
    """A pattern that matches another pattern with a certain type annotation.

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern that needs type annotation.

    ttype: tvm.ir.type.Type
        The type to match.
    """

    def __init__(self, pattern: "DFPattern", ttype: tvm.ir.type.Type):
        self.__init_handle_by_constructor__(ffi.TypePattern, pattern, ttype)


@register_df_node
class DataTypePattern(DFPattern):
    """A pattern that matches another pattern with certain data type

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern that needs type annotation.

    dtype: str
        The dtype to match.
    """

    def __init__(self, pattern: "DFPattern", dtype: str):
        self.__init_handle_by_constructor__(ffi.DataTypePattern, pattern, dtype)


@register_df_node
class ShapePattern(DFPattern):
    """A pattern that matches another pattern with a certain tensor shape

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern that needs type annotation.

    shape: List[tvm.ir.PrimExpr]
        The shape to match.
    """

    def __init__(self, pattern: "DFPattern", shape: List[tvm.ir.PrimExpr]):
        self.__init_handle_by_constructor__(ffi.ShapePattern, pattern, shape)


@register_df_node
class PrimArrPattern(DFPattern):
    def __init__(self, shape: List[tvm.ir.PrimExpr]):
        self.__init_handle_by_constructor__(ffi.PrimArrPattern, shape)


@register_df_node
class AttrPattern(DFPattern):
    """Get match an expression with a certain attributes.
    Currently only supports Op Attributes, not call Attributes.

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern.

    attrs: tvm.ir.attrs.Attrs
        The attributes to match.
    """

    def __init__(self, pattern: "DFPattern", attrs: tvm.ir.attrs.Attrs):
        self.__init_handle_by_constructor__(ffi.AttrPattern, pattern, attrs)


def is_var(name: str = "") -> "DFPattern":
    """
    Syntatic sugar for creating an optionally named VarPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return VarPattern(name)


def is_gv(name: str = "") -> "DFPattern":
    return GlobalVarPattern(name)


def is_dfv(name: str = "") -> "DFPattern":
    return DataflowVarPattern(name)


def is_const() -> "DFPattern":
    """
    Syntatic sugar for creating a ConstantPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return ConstantPattern()


def is_expr(expr: Expr) -> "DFPattern":
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    expr: Expr
        The Relax expression to match.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return ExprPattern(expr)


def is_op(op_name: str) -> "DFPattern":
    """
    Syntatic sugar for creating an operator ExprPattern.

    Parameters
    ----------
    op_name: String
        The name of the tvm.ir.op.Op object

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting ExprPattern
    """
    op = get(op_name)
    return ExprPattern(op)


def is_tuple(fields: tvm.ir.container.Array, unordered=False) -> "DFPattern":
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    fields : Array[tvm.relax.dataflow_pattern.DFPattern]
        The fields in the tuple.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    if unordered:
        return UnorderedTuplePattern(fields)
    return TuplePattern(fields)


def is_tuple_get_item(tuple_value: "DFPattern", index: Optional[int] = None) -> "DFPattern":
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    tuple_value: tvm.relax.dataflow_pattern.DFPattern
        The input tuple expression.

    index: Optional[int]
        The index to match; Default (None) to match a TupleGetItem with any index.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return TupleGetItemPattern(tuple_value, index)


def wildcard() -> "DFPattern":
    """
    Syntatic sugar for creating a WildcardPattern.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return WildcardPattern()


def has_type(ttype: tvm.ir.type.Type, pattern: "DFPattern" = None) -> "DFPattern":
    """
    Syntatic sugar for creating a TypePattern

    Parameters
    ----------
    ttype: tvm.ir.type.Type
        The type to match

    pattern: tvm.relax.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting TypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return TypePattern(pattern, ttype)


def has_dtype(dtype: str, pattern: "DFPattern" = None) -> "DFPattern":
    """
    Syntatic sugar for creating a DataTypePattern

    Parameters
    ----------
    dtype: str
        The dtype to match

    pattern: tvm.relax.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting DataTypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return DataTypePattern(pattern, dtype)


def has_shape(shape: List[tvm.ir.PrimExpr], pattern: "DFPattern" = None) -> "DFPattern":
    """Either has_shape(a, b, c) or has_shape([a, b, c, ...])"""
    if not isinstance(shape, (list, tuple, tvm.ir.PrimExpr)):
        raise ValueError("has_shape takes a list or tuple as input.")
    if pattern is None:
        pattern = wildcard()
    return ShapePattern(pattern, shape)


def is_shape(shape: List[tvm.ir.PrimExpr]) -> "DFPattern":
    if not isinstance(shape, (list, tuple, tvm.ir.PrimExpr)):
        raise ValueError("is_shape takes a list or tuple as input.")
    return PrimArrPattern(shape)


def is_call_tir(
    func_name: str,
    args: Union[List, Tuple, TuplePattern] = None,
    shape: Union[Tuple, List[tvm.ir.PrimExpr], DFPattern] = None,
) -> "DFPattern":
    if args is None:
        args = wildcard()
    elif isinstance(args, (list, tuple)):
        args = TuplePattern(args)

    if shape is None:
        shape = wildcard()
    elif isinstance(shape, (list, tuple, tvm.ir.container.Array)):
        shape = is_tuple(shape)  # multiple shape patterns

    return is_op("relax.call_tir")(GlobalVarPattern(func_name), args, shape)


def deny(pattern: "DFPattern") -> "DFPattern":
    return NotPattern(pattern)


def has_rt_dep_shape():
    return RuntimeDepShapePattern()


def has_attr(attrs, pattern=None) -> "DFPattern":
    """
    Syntatic sugar for creating an AttrPattern

    Parameters
    ----------
    attrs: Dict[str, Object]
        The attributes to match

    pattern: Optional[tvm.relax.dataflow_pattern.DFPattern]
        The input pattern.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting AttrPattern
    """
    if pattern is None:
        pattern = wildcard()
    return pattern.has_attr(attrs)


def match_expr(pattern: "DFPattern", expr: Expr, var2val: Dict[Var, Expr] = None) -> bool:
    """
    Match a pattern to an expression

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern.
    expr : tvm.relax.Expr
        The expression to match.
    var2val : Optional[Dict[tvm.relax.Var, tvm.relax.Expr]]
        A mapping from variables to values for autojump.
    """
    return ffi.match_expr(pattern, expr, var2val)


@register_df_node
class UsedBySeq(Node):
    """A sequence of patterns that patterns[i] is used by patterns[i+1]"""

    def __init__(self, patterns: List[DFPattern]):
        """Create a chain that a pattern is used by its follower.

        Args:
            patterns (List[DFPattern]): patterns with used-by relation.
        """
        self.__init_handle_by_constructor__(ffi.UsedBySeq, patterns)

    def used_by(self, other: Union[DFPattern, "UsedBySeq"], index=-1) -> "UsedBySeq":
        return used_by(self, other, index)

    def __xor__(self, other) -> "UsedBySeq":
        return self.used_by(other, -1)

    def dup(self) -> "UsedBySeq":
        return ffi.dup_ubseq(self)


@register_df_node
class OnlyUsedBySeq(Node):
    """A sequence of patterns that patterns[i] is ONLY used by patterns[i+1]"""

    def __init__(self, patterns: List[DFPattern]):
        """Create a chain that a pattern is only used by its follower.

        Args:
            patterns (List[DFPattern]): patterns with only-used-by relation.
        """
        self.__init_handle_by_constructor__(ffi.OnlyUsedBySeq, patterns)

    def only_used_by(self, other: Union[DFPattern, "OnlyUsedBySeq"], index=-1) -> "OnlyUsedBySeq":
        return only_used_by(self, other, index)

    def __rshift__(self, other) -> "OnlyUsedBySeq":
        return self.only_used_by(other, -1)

    def dup(self) -> "OnlyUsedBySeq":
        return ffi.dup_oubseq(self)


def match_dfb(
    ctx: "PatternContext",
    dfb: DataflowBlock,
    start_hint: Optional[Var] = None,
    match_once: bool = False,
) -> Dict[DFPattern, Var]:
    """
    Match a pattern to a function

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern with graph constraints.
    expr : tvm.relax.Function
        The function to match.
    """
    if ctx is None:
        ctx = PatternContext.current()
    return ffi.match_dfb(ctx, dfb, start_hint, match_once)


def used_by(
    lhs: Union[DFPattern, UsedBySeq, OnlyUsedBySeq],
    rhs: Union[DFPattern, UsedBySeq, OnlyUsedBySeq],
    index=-1,
) -> UsedBySeq:
    if isinstance(lhs, DFPattern):
        lhs_ = UsedBySeq([lhs])
    elif isinstance(lhs, OnlyUsedBySeq):
        lhs_ = UsedBySeq([lhs.patterns[-1]])
    else:
        lhs_ = lhs

    if isinstance(rhs, DFPattern):
        rhs_ = UsedBySeq([rhs])
    elif isinstance(rhs, OnlyUsedBySeq):
        rhs_ = UsedBySeq([rhs.patterns[0]])
    else:
        rhs_ = rhs

    return ffi.used_by(lhs_, rhs_, index)


def only_used_by(
    lhs: Union[DFPattern, OnlyUsedBySeq], rhs: Union[DFPattern, OnlyUsedBySeq], index=-1
) -> OnlyUsedBySeq:
    if isinstance(lhs, DFPattern):
        lhs = OnlyUsedBySeq([lhs])
    if isinstance(rhs, DFPattern):
        rhs = OnlyUsedBySeq([rhs])
    return ffi.only_used_by(lhs, rhs, index)


def dup(*args):
    if len(args) == 1:
        if isinstance(args[0], (list, tuple, tvm.ir.container.Array)):
            return [x.dup() for x in args[0]]
        return args[0].dup()
    return tuple([v.dup() for v in args])


def fork_to(*args):
    root_pattern = wildcard()
    for arg in args:
        root_pattern ^ arg


class PatternContext(tvm.runtime.Object):
    """A context object for doing graph (topogical) pattern matching."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.PatternContext)

    def __enter__(self):
        ffi.enter_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ffi.exit_context(self)

    @staticmethod
    def current():
        return ffi.current_context()

    def match_dfb(
        self, dfb: DataflowBlock, start_hint: Optional[Var] = None, match_once: bool = False
    ) -> Dict[DFPattern, Var]:
        return match_dfb(self, dfb, start_hint, match_once)
