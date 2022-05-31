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
from typing import List, Optional

import tvm._ffi
from tvm.relax import Expr
from tvm.relay.op import get
import tvm.relay.dataflow_pattern as relay_dp
from tvm.relay.dataflow_pattern import register_df_node as register_relay_df_node


from . import _ffi as ffi

def register_relax_df_node(type_key=None):
    """Register a Relax node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm._ffi.register_object("relax.dataflow_pattern." + type_key.__name__)(type_key)
    return tvm._ffi.register_object(type_key)


class DFPattern(relay_dp.DFPattern):
    """Base class of all Relax Patterns."""

    def match(self, expr: Expr) -> bool:
        """
        Match this pattern to an expression

        Parameters
        ----------
        expr : tvm.relax.Expr
            The expression to match.

        Returns
        -------
        result: bool
            Whether or not the expression matches the pattern
        """
        return match(self, expr)


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


def is_constant() -> "DFPattern":
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
        The name of the relay op

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting ExprPattern
    """
    op = get(op_name)
    return ExprPattern(op)


def is_tuple(fields: tvm.ir.container.Array) -> "DFPattern":
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


def is_if(cond, true_branch, false_branch):
    """
    Syntatic sugar for creating an IfPattern.

    Parameters
    ----------
    cond: tvm.relax.dataflow_pattern.DFPattern
        The pattern describing the condition of If.

    true_branch: tvm.relax.dataflow_pattern.DFPattern
        The pattern describing the true branch of If.

    false_branch: tvm.relax.dataflow_pattern.DFPattern
        The pattern describing the false branch of If.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return IfPattern(cond, true_branch, false_branch)


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
    """
    Syntatic sugar for creating a ShapePattern

    Parameters
    ----------
    shape: List[tvm.ir.PrimExpr]
        The shape to match

    pattern: tvm.relax.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting ShapePattern
    """
    if pattern is None:
        pattern = wildcard()
    return ShapePattern(pattern, shape)


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


def dominates(parent: "DFPattern", path: "DFPattern", child: "DFPattern") -> "DFPattern":
    """
    Syntatic sugar for creating an Dominator pattern

    Parameters
    ----------
    parent: tvm.relax.dataflow_pattern.DFPattern
        The parent pattern.
    path: tvm.relax.dataflow_pattern.DFPattern
        The fuzzy path pattern.
    child: tvm.relax.dataflow_pattern.DFPattern
        The child pattern.

    Returns
    -------
    result: tvm.relax.dataflow_pattern.DFPattern
        The resulting DominatorPattern.
    """
    return DominatorPattern(parent, path, child)


def match(pattern: "DFPattern", expr: Expr) -> bool:
    """
    Match a pattern to an expression

    Parameters
    ----------
    pattern: tvm.relax.dataflow_pattern.DFPattern
        The input pattern.
    expr : tvm.relax.Expr
        The expression to match.
    """
    return ffi.match(pattern, expr)


@register_relax_df_node
class RuntimeDepShapePattern(DFPattern):
    """A pattern matching a Relax RuntimeDepShape."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.RuntimeDepShapePattern)


@register_relay_df_node
class ExprPattern(DFPattern):
    """A pattern which matches a constant expression.

    Parameters
    ----------
    expr : tvm.relax.Expr
        The expression to match.
    """

    def __init__(self, expr: Expr):
        self.__init_handle_by_constructor__(ffi.ExprPattern, expr)


@register_relay_df_node
class VarPattern(DFPattern):
    """A local variable pattern.

    Local variable can be used to declare input
    arguments to a function, or intermediate variables.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any VarNode.

    type_annotation: tvm.ir.type.Type, optional
        The type annotation on the variable.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.VarPattern, name_hint)


@register_relay_df_node
class ConstantPattern(DFPattern):
    """A pattern matching a Relax Constant."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.ConstantPattern)


@register_relay_df_node
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


@register_relay_df_node
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


@register_relay_df_node
class IfPattern(DFPattern):
    """A patern matching a Relax If.

    Parameters
    ----------
    cond: tvm.relax.dataflow_pattern.DFPattern
        The pattern describing the condition of If.

    true_branch: tvm.relax.dataflow_pattern.DFPattern
        The pattern describing the true branch of If.

    false_branch: tvm.relax.dataflow_pattern.DFPattern
        The pattern describing the false branch of If.
    """

    def __init__(self, cond: "DFPattern", true_branch: "DFPattern", false_branch: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.IfPattern, cond, true_branch, false_branch)


@register_relay_df_node
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


@register_relay_df_node
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


@register_relay_df_node
class AltPattern(DFPattern):
    """Create a Pattern that can match one of two conditions

    Parameters
    ----------
    left: tvm.relax.dataflow_pattern.DFPattern
        One possible matching pattern.
    right: tvm.relax.dataflow_pattern.DFPattern
        One possible matching pattern.
    """

    def __init__(self, left: "DFPattern", right: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.AltPattern, left, right)


@register_relay_df_node
class WildcardPattern(DFPattern):
    """A pattern which matches anything."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.WildcardPattern)


@register_relay_df_node
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


@register_relay_df_node
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


@register_relay_df_node
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


@register_relay_df_node
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


@register_relay_df_node
class DominatorPattern(DFPattern):
    """Match a domination graph.

    Parameters
    ----------
    parent: tvm.relax.dataflow_pattern.DFPattern
        The parent, i.e., the single node which produces something,
        later aggregated by the child.
    path: tvm.relax.dataflow_pattern.DFPattern
        The fuzzy path pattern between parent and child,
        typically matches elementwise ops.
    child: tvm.relax.dataflow_pattern.DFPattern
        The last node in the domination which is the end user
        for all nodes in the path and the parent.
    """

    def __init__(self, parent: "DFPattern", path: "DFPattern", child: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.DominatorPattern, parent, path, child)

