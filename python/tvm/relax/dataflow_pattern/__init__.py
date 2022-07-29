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
# pylint: disable=pointless-statement

from typing import List, Optional, Dict, Union, Tuple

import tvm
import tvm._ffi as tvm_ffi
from tvm.ir.expr import PrimExpr
from tvm.relax import DataflowBlock, Expr, Var
from tvm.relay.op import get
from tvm.ir.container import Array

from ...ir import make_node
from ...runtime import Object
from ...ir.base import Node
from . import _ffi as ffi


def register_df_node(type_key=None):
    """
    Register a Relax node type

    Parameters
    ----------
    type_key : str or cls
        The type key of the node
    """
    if not isinstance(type_key, str):
        return tvm_ffi.register_object("relax.dataflow_pattern." + type_key.__name__)(type_key)
    return tvm_ffi.register_object(type_key)


class DFPattern(Node):
    """Base class of all Patterns."""

    def __call__(self, *args, varg_default_wildcard=False) -> "CallPattern":
        """
        Syntax sugar for creating a CallPattern with argument patterns

        Returns
        -------
        result: CallPattern
            The resulting CallPattern
        """
        return CallPattern(self, args, varg_default_wildcard)

    def __or__(self, other: "DFPattern") -> "OrPattern":
        """
        Syntax sugar for creating an OrPattern

        Parameters
        ----------
        other: DFPattern
            Alternative pattern

        Returns
        -------
        result: OrPattern
            The resulting OrPattern
        """
        return OrPattern(self, other)

    def __and__(self, other: "DFPattern") -> "AndPattern":
        """
        Syntax sugar for creating an AndPattern

        Parameters
        ----------
        other: DFPattern
            Additional pattern to satisfy

        Returns
        -------
        result: AndPattern
            The resulting AndPattern
        """
        return AndPattern(self, other)

    def __add__(self, other: "DFPattern") -> "CallPattern":
        """
        Syntax sugar for creating a relax.add CallPattern

        Parameters
        ----------
        other: DFPattern
            DFPattern representing a relax.Var to add

        Returns
        -------
        result: CallPattern
            The resulting CallPattern
        """
        return is_op("relax.add")(self, other)

    def __sub__(self, other: "DFPattern") -> "CallPattern":
        """
        Syntax sugar for creating a relax.subtract CallPattern

        Parameters
        ----------
        other: DFPattern
            DFPattern representing a relax.Var to subtract

        Returns
        -------
        result: CallPattern
            The resulting CallPattern
        """
        return is_op("relax.subtract")(self, other)

    def __mul__(self, other: "DFPattern") -> "CallPattern":
        """
        Syntax sugar for creating a relax.multiply CallPattern

        Parameters
        ----------
        other: DFPattern
            DFPattern representing a relax.Var to multiply

        Returns
        -------
        result: CallPattern
            The resulting CallPattern
        """
        return is_op("relax.multiply")(self, other)

    def __truediv__(self, other: "DFPattern") -> "CallPattern":
        """
        Syntax sugar for creating a relax.divide CallPattern

        Parameters
        ----------
        other: DFPattern
            DFPattern representing a relax.Var to divide

        Returns
        -------
        result: CallPattern
            The resulting CallPattern
        """
        return is_op("relax.divide")(self, other)

    def __invert__(self) -> "NotPattern":
        """
        Syntax sugar for creating a DFPattern to reject

        Returns
        -------
        result: NotPattern
            The resulting NotPattern
        """
        return reject(self)

    def has_attr(self, attrs: Dict[str, Object]) -> "AttrPattern":
        """
        Add an attribute constraint to this pattern

        Parameters
        ----------
        attrs: Dict[str, Object]

        Returns
        -------
        result: AttrPattern
            The resulting AttrPattern
        """
        attrs = make_node("DictAttrs", **attrs)
        return AttrPattern(self, attrs)

    def has_type(self, ttype: tvm.ir.type.Type) -> "TypePattern":
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        ttype: tvm.ir.type.Type
            The type to match

        Returns
        -------
        result: TypePattern
            The resulting TypePattern
        """
        return TypePattern(self, ttype)

    def has_dtype(self, dtype: str) -> "DataTypePattern":
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        dtype: str
            The dtype to match

        Returns
        -------
        result: DataTypePattern
            The resulting DataTypePattern
        """
        return has_dtype(dtype, self)

    def has_shape(self, shape: List[PrimExpr]) -> "ShapePattern":
        """
        Add a shape constraint to this pattern

        Parameters
        ----------
        shape: List[PrimExpr]
            Expected shape list

        Returns
        -------
        result: ShapePattern
            The resulting ShapePattern

        Notes
        -----
        has_shape assumes that the matched relax.Expr only has one
        output tensor. Use is_tuple for those with multiple outputs.
        """
        if not isinstance(shape, (list, tuple, tvm.ir.PrimExpr)):
            raise ValueError("has_shape takes a list or tuple as input.")
        return ShapePattern(pattern=self, shape=shape)

    def match(self, expr, var2val: Optional[Dict[Var, Expr]] = None) -> bool:
        """
        Match a relax.Expr syntactically

        Parameters
        ----------
        expr : tvm.relax.Expr
            The expression to match
        var2val : Optional[Dict[tvm.relax.Var, tvm.relax.Expr]]
            A mapping from relax.Var to relax.Expr for autojump.

        Returns
        -------
        result: bool
            Whether or not the expression matches the pattern

        Notes
        -----
        Unlike Relay whose function is an expression, functions in Relax consists
        of blocks of bindings that they are not syntactically connected. We use a
        mapping (i.e., var2val) to migrate the gap. For example, to when matching
        "relax.add(lv0, lv1)", given var2val, we match lv0's binded expression
        when the recursive pattern matching goes to check lv0. The var2val mapping
        can be computed through the tvm.relax.analysis.get_var2val function.
        """
        return match_expr(self, expr, var2val)

    def has_rt_dep_shape(self) -> "AndPattern":
        """
        Syntax sugar for assuming current node has a runtime-dependent shape

        Returns
        -------
        result: AndPattern
            The resulting AndPattern
        """
        return RuntimeDepShapePattern(self)

    def used_by(self, other: Union["DFPattern", "PatternSeq"], index=-1) -> "PatternSeq":
        """
        The current pattern being used by another pattern (sequence)

        Parameters
        ----------
        other : Union[DFPattern, DFPattern]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        result: PatternSeq
            A chained pattern sequence
        """
        return _used_by(self, other, index)

    def __xor__(self, other: Union["DFPattern", "PatternSeq"]) -> "PatternSeq":
        """Syntax sugar of DFPattern.used_by"""
        return self.used_by(other, -1)

    def only_used_by(self, other: Union["DFPattern", "PatternSeq"], index=-1) -> "PatternSeq":
        """
        The current pattern being **ONLY** used by another pattern (sequence)

        Parameters
        ----------
        other : Union[DFPattern, DFPattern]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        result: PatternSeq
            A chained pattern sequence
        """
        return _only_used_by(self, other, index)

    def __rshift__(self, other: Union["DFPattern", "PatternSeq"]) -> "PatternSeq":
        """Syntax sugar of DFPattern.only_used_by"""
        return self.only_used_by(other, -1)

    def dup(self) -> "DFPattern":
        """
        Duplicate the current pattern (new object under different address)

        Returns
        -------
        DFPattern
            A duplicated pattern
        """
        return ffi.dup_pattern(self)

    def fork_to(self, *args) -> None:
        """Fork the current pattern to multiple pattern branches"""
        for v in args:
            self ^ v


@register_df_node
class RuntimeDepShapePattern(DFPattern):
    """A pattern matching a Relax RuntimeDepShape."""

    def __init__(self, pattern: DFPattern):
        self.__init_handle_by_constructor__(ffi.RuntimeDepShapePattern, pattern)


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

    varg_default_wildcard: bool
        If True, args can be fewer than actual provided arguments.

    Notes
    -----
    By setting varg_default_wildcard to True, we can only focus on the argument
    patterns we specified. For example, CallPattern(Op, [A, B]) can match
    a call of Op(A, B) or Op(A, B, C, ...) that has more arguments. However,
    the specified argument patterns must be matched (i.e., A and B).
    """

    def __init__(
        self,
        op: "DFPattern",
        args: List["DFPattern"],
        varg_default_wildcard: bool = False,
    ):
        self.__init_handle_by_constructor__(ffi.CallPattern, op, args, varg_default_wildcard)


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

    def __init__(self, fields: Array):
        self.__init_handle_by_constructor__(ffi.TuplePattern, fields)

    def __getitem__(self, index: int) -> "TupleGetItemPattern":
        if index >= len(self):
            raise IndexError("TuplePattern index out of range")
        return TupleGetItemPattern(self, index)

    def __len__(self):
        return len(self.fields)


@register_df_node
class UnorderedTuplePattern(DFPattern):
    """A patern matching a Relax Tuple unorderedly.

    Parameters
    ----------
    fields : Array[tvm.relax.dataflow_pattern.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields: Array):
        self.__init_handle_by_constructor__(ffi.UnorderedTuplePattern, fields)

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
    """
    A pattern to match an array of PrimExpr

    Parameters
    ----------
    shape : List[tvm.ir.PrimExpr]
        The shape to match.
    """

    def __init__(self, shape: List[tvm.ir.PrimExpr]):
        self.__init_handle_by_constructor__(ffi.PrimArrPattern, shape)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("PrimArrPattern index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)


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


def is_var(name: str = "") -> VarPattern:
    """
    Syntatic sugar for creating an optionally named VarPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.VarPattern
        The resulting pattern.
    """
    return VarPattern(name)


def is_gv(name: str = "") -> GlobalVarPattern:
    """Syntax sugar for creating an optionally (if name is empty) named GlobalVarPattern."""
    return GlobalVarPattern(name)


def is_dfv(name: str = "") -> DataflowVarPattern:
    """Syntax sugar for creating an optionally (if name is empty) named DataflowVarPattern."""
    return DataflowVarPattern(name)


def is_const() -> ConstantPattern:
    """
    Syntatic sugar for creating a ConstantPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.ConstantPattern
        The resulting pattern.
    """
    return ConstantPattern()


def is_expr(expr: Expr) -> ExprPattern:
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    expr: Expr
        The Relax expression to match.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.ExprPattern
        The resulting pattern.
    """
    return ExprPattern(expr)


def is_op(op_name: str) -> ExprPattern:
    """
    Syntatic sugar for creating an operator ExprPattern.

    Parameters
    ----------
    op_name: String
        The name of the tvm.ir.op.Op object

    Returns
    -------
    result: tvm.relax.dataflow_pattern.ExprPattern
        The resulting ExprPattern
    """
    op = get(op_name)
    return ExprPattern(op)


def is_tuple(
    fields: Union[Array, List, Tuple], unordered=False
) -> Union[TuplePattern, UnorderedTuplePattern]:
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
    if not isinstance(fields, (list, tuple, Array)):
        raise ValueError("fields must be a list, tuple, or Array")
    if unordered:
        return UnorderedTuplePattern(fields)
    return TuplePattern(fields)


def is_tuple_get_item(tuple_value: DFPattern, index: Optional[int] = None) -> TupleGetItemPattern:
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
    result: tvm.relax.dataflow_pattern.TupleGetItemPattern
        The resulting pattern.
    """
    return TupleGetItemPattern(tuple_value, index)


def wildcard() -> WildcardPattern:
    """
    Syntatic sugar for creating a WildcardPattern.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.WildcardPattern
        The resulting pattern.
    """
    return WildcardPattern()


def has_dtype(dtype: str, pattern: DFPattern = None) -> DataTypePattern:
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
    result: tvm.relax.dataflow_pattern.DataTypePattern
        The resulting DataTypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return DataTypePattern(pattern, dtype)


def is_shape(shape: List[tvm.ir.PrimExpr]) -> "PrimArrPattern":
    if not isinstance(shape, (list, tuple, tvm.ir.PrimExpr)):
        raise ValueError("is_shape takes a list or tuple as input.")
    return PrimArrPattern(shape)


def is_call_tir(
    func_name: str,
    args: Union[List, Tuple, TuplePattern] = None,
    shape: Union[Tuple, List[tvm.ir.PrimExpr], DFPattern] = None,
) -> CallPattern:
    """
    Syntax sugar for creating a CallPattern for call_tir

    Parameters
    ----------
    func_name : str
        Name of the CPS function to call.
    args : Union[List[DFPattern], Tuple[DFPattern]], optional
        Arguments in expected call_packed, by default None meaning arbitrary (number of) arguments
    shape : Union[Tuple, List[tvm.ir.PrimExpr], DFPattern], optional
        Shape (or shapes in a tuple) of the output, by default None meaning arbitrary shape(s)

    Returns
    -------
    CallPattern
        The resulting CallPattern
    """
    if args is None:
        args = wildcard()
    elif isinstance(args, (list, tuple)):
        args = TuplePattern(args)

    if shape is None:
        shape = wildcard()
    elif isinstance(shape, (list, Array)):
        shape = PrimArrPattern(shape)
    elif isinstance(shape, (tuple)):
        shape = is_tuple(shape)  # multiple shape patterns

    return is_op("relax.call_tir")(GlobalVarPattern(func_name), args, shape)


def is_call_packed(
    func_name: str, args: Union[List[DFPattern], Tuple[DFPattern]] = None
) -> CallPattern:
    """
    Syntax sugar for creating a CallPattern for call_packed

    Parameters
    ----------
    func_name : str
        Name of the external function to call
    args : Union[List[DFPattern], Tuple[DFPattern]], optional
        Arguments in expected call_packed, by default None meaning arbitrary (number of) arguments

    Returns
    -------
    CallPattern
        The resulting CallPattern
    """
    if args is None:
        return ExternFuncPattern(func_name)(varg_default_wildcard=True)
    return ExternFuncPattern(func_name)(*args)


def reject(pattern: DFPattern) -> NotPattern:
    """
    Syntax sugar for creating a DFPattern to reject

    Parameters
    ----------
    pattern : DFPattern
        The pattern to deny

    Returns
    -------
    result: NotPattern
        The resulting NotPattern
    """
    return NotPattern(pattern)


def has_attr(attrs, pattern=None) -> AttrPattern:
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
class PatternSeq(Node):
    """A sequence of patterns with consecutive constraints"""

    def __init__(self, patterns: List[DFPattern], only_use=False):
        """
        Initializer to PatternSeq

        Parameters
        ----------
        patterns : List[DFPattern]
            A chain of patterns
        only_use : bool, optional
            Whether the patterns follows only-used-by relations consecutively, by default False
        """
        self.__init_handle_by_constructor__(ffi.PatternSeq, patterns, only_use)

    def used_by(self, other: Union[DFPattern, "PatternSeq"], index=-1) -> "PatternSeq":
        """
        Assuming the right-most pattern must be used by the `other` pattern as a producer

        Parameters
        ----------
        other : Union[DFPattern, PatternSeq]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        PatternSeq
            A chained pattern sequence

        Notes
        -----
        If other is PatternSeq, it means the right-most pattern must be used by the left-most
        pattern of the other sequence.
        """
        return _used_by(self, other, index)

    def only_used_by(self, other: Union[DFPattern, "PatternSeq"], index=-1) -> "PatternSeq":

        """
        Assuming the right-most pattern must be **ONLY** used by the `other` pattern as a producer

        Parameters
        ----------
        other : Union[DFPattern, PatternSeq]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        PatternSeq
            A chained pattern sequence

        Notes
        -----
        If other is PatternSeq, it means the right-most pattern must be **ONLY** used by the
        left-most pattern of the other sequence.
        """
        return _only_used_by(self, other, index)

    def __getitem__(self, index: int) -> DFPattern:
        """
        Access the pattern at the given index

        Parameters
        ----------
        index : int
            Index of the accessed pattern

        Returns
        -------
        DFPattern
            The accessed pattern
        """
        return self.patterns[index]

    def __xor__(self, other) -> "PatternSeq":
        """Syntax sugar of PatternSeq.used_by"""
        return self.used_by(other, -1)

    def __rshift__(self, other) -> "PatternSeq":
        """Syntax sugar of PatternSeq.only_used_by"""
        return self.only_used_by(other, -1)

    def dup(self) -> "PatternSeq":
        """
        Duplicate the pattern sequence (new object under different address)

        Returns
        -------
        PatternSeq
            A duplicated chain
        """
        return ffi.dup_seq(self)


def match_dfb(
    ctx: "PatternContext",
    dfb: DataflowBlock,
    start_hint: Optional[Var] = None,
    must_include_hint: bool = False,
) -> Dict[DFPattern, Var]:
    """
    Match a DataflowBlock via a graph of DFPattern and corresponding constraints

    Parameters
    ----------
    ctx : PatternContext
        The object to store graph-matching context (e.g., edge constraints)
    dfb : DataflowBlock
        The DataflowBlock to match
    start_hint : Optional[Var], optional
        Indicating the starting expression to match, by default None
    must_include_hint : bool, optional
        Whether the start_hint expression must be matched, by default False

    Returns
    -------
    Dict[DFPattern, Var]
        The mapping from DFPattern to matched expression
    """
    if ctx is None:
        ctx = PatternContext.current()
    return ffi.match_dfb(ctx, dfb, start_hint, must_include_hint)


class PatternContext(tvm.runtime.Object):
    """A context object for doing graph (topogical) pattern matching."""

    def __init__(self):
        """
        Initialize the PatternContext
        """
        self.__init_handle_by_constructor__(ffi.PatternContext)

    def __enter__(self):
        """Enter the context"""
        ffi.enter_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context"""
        ffi.exit_context(self)

    @staticmethod
    def current() -> "PatternContext":
        """
        Get the current context

        Returns
        -------
        PatternContext
            The current context
        """
        return ffi.current_context()

    def match_dfb(
        self, dfb: DataflowBlock, start_hint: Optional[Var] = None, must_include_hint: bool = False
    ) -> Dict[DFPattern, Var]:
        """
        Match a DataflowBlock via a graph of DFPattern and corresponding constraints

        Parameters
        ----------
        dfb : DataflowBlock
            The DataflowBlock to match
        start_hint : Optional[Var], optional
            Indicating the starting expression to match, by default None
        must_include_hint : bool, optional
            Whether the start_hint expression must be matched, by default False

        Returns
        -------
        Dict[DFPattern, Var]
            The mapping from DFPattern to matched expression
        """
        return match_dfb(self, dfb, start_hint, must_include_hint)


### Private functions


def _used_by(
    lhs: Union[DFPattern, PatternSeq],
    rhs: Union[DFPattern, PatternSeq],
    index=-1,
) -> PatternSeq:
    if isinstance(lhs, DFPattern):
        lhs = PatternSeq([lhs])
    if isinstance(rhs, DFPattern):
        rhs = PatternSeq([rhs])
    return ffi.used_by(lhs, rhs, index)


def _only_used_by(
    lhs: Union[DFPattern, PatternSeq], rhs: Union[DFPattern, PatternSeq], index=-1
) -> PatternSeq:
    if isinstance(lhs, DFPattern):
        lhs = PatternSeq([lhs])
    if isinstance(rhs, DFPattern):
        rhs = PatternSeq([rhs])
    return ffi.only_used_by(lhs, rhs, index)
