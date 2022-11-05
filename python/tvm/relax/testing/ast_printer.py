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
# pylint: disable=redefined-builtin, abstract-method, arguments-differ
"""
Utility script for printing Relax modules as AST diagrams,
only intended to show how the AST is put together.
It is not a pretty-printer and, in fact, is more of an ugly-printer,
but it can be useful for tutorials and debugging.
"""
from __future__ import annotations  # must import to defer parsing of annotations
from typing import Iterable
import tvm
from tvm import relax
from tvm.ir.expr import PrimExpr
from tvm.relax import ExprFunctor


def wrap_quotes(text: str) -> str:
    """
    Wraps the text in quotes.
    """
    return f'"{text}"'


class ASTPrinter(ExprFunctor):
    """
    Class for recursing down ASTs and printing them in a very simple format,
    mainly for instructive purposes and, perhaps, debugging.
    """

    def __init__(
        self,
        indent_str="    ",
        include_type_annotations=True,
        include_shape_annotations=True,
        include_call_attrs=True,
    ):
        self.indent_str = indent_str
        self.include_type_annotations = include_type_annotations
        self.include_shape_annotations = include_shape_annotations
        self.include_call_attrs = include_call_attrs

    def visit_expr(self, expr: relax.Expr) -> str:
        # extend so we also dispatch to bindings and binding blocks,
        # a little silly but IRFunctor hasn't been ported to Python
        if isinstance(expr, relax.DataflowBlock):
            return self.visit_dataflow_block_(expr)
        if isinstance(expr, relax.BindingBlock):
            return self.visit_binding_block_(expr)
        if isinstance(expr, relax.Binding):
            return self.visit_binding_(expr)
        return super().visit_expr(expr)

    def indent(self, text: str) -> str:
        """
        Indent all lines of the input.
        """
        if text == "":
            return ""
        lines = text.split("\n")
        return self.indent_str + f"\n{self.indent_str}".join(lines)

    def build_ast_node(self, nodename: str, force_newline=False, **kwargs: str) -> str:
        """
        Returns 'nodename(..., fields[i][0]=fields[i][1], ...)'
        with appropriate indentation
        """
        return self.build_list(
            map(lambda field: f"{field[0]}={field[1]}", kwargs.items()),
            open_tok=f"{nodename}(",
            close_tok=")",
            force_newline=force_newline,
        )

    def build_list(
        self, members: Iterable[str], open_tok="[", close_tok="]", force_newline=False
    ) -> str:
        """
        Builds a list of the members given, appropriately indented,
        with each field on a line.
        (special case: if there is only one field, then we do not put it on a new line
        unless that field contains a newline or `force_newline` is set to true).
        `open_tok` and `close_tok` are used to open and close the list, respectively.
        """
        mem_list = list(members)
        if not mem_list:
            return f"{open_tok}{close_tok}"
        if len(mem_list) == 1 and not force_newline and "\n" not in mem_list[0]:
            return f"{open_tok}{mem_list[0]}{close_tok}"
        member_lines = ",\n".join(map(self.indent, mem_list))
        return f"{open_tok}\n{member_lines}\n{close_tok}"

    def visit_constant_(self, op: relax.Constant) -> str:
        # simple rule of thumb: keep scalars inline, but anything larger goes on a new one
        force_newline = len(op.data.shape) > 0
        return self.build_ast_node("Constant", force_newline=force_newline, data=str(op.data))

    def visit_tuple_(self, op: relax.Tuple) -> str:
        return self.build_ast_node("Tuple", fields=self.build_list(map(self.visit_expr, op.fields)))

    def visit_dataflow_var_(self, op: relax.DataflowVar) -> str:
        fields = {"name_hint": wrap_quotes(op.name_hint)}
        if op.shape_ and self.include_shape_annotations:
            fields["shape_"] = self.visit_expr(op.shape_)
        if op._checked_type_ and self.include_type_annotations:
            fields["_checked_type_"] = self.visit_type_(op._checked_type_)
        return self.build_ast_node("DataflowVar", **fields)

    def visit_var_(self, op: relax.Var) -> str:
        fields = {"name_hint": wrap_quotes(op.name_hint)}
        if op.shape_ and self.include_shape_annotations:
            fields["shape_"] = self.visit_expr(op.shape_)
        if op._checked_type_ and self.include_type_annotations:
            fields["_checked_type_"] = self.visit_type_(op._checked_type_)
        return self.build_ast_node("Var", **fields)

    def visit_shape_expr_(self, op: relax.ShapeExpr) -> str:
        return self.build_ast_node(
            "ShapeExpr", values=self.build_list(map(self.visit_prim_expr_, op.values))
        )

    def visit_runtime_dep_shape_(self, _: relax.RuntimeDepShape) -> str:
        # no fields, apparently?
        return self.build_ast_node("RuntimeDepShape")

    def visit_extern_func_(self, op: relax.ExternFunc) -> str:
        return self.build_ast_node("ExternFunc", global_symbol=wrap_quotes(op.global_symbol))

    def visit_global_var_(self, op: relax.GlobalVar) -> str:
        return self.build_ast_node("GlobalVar", name_hint=wrap_quotes(op.name_hint))

    def visit_function_(self, op: relax.Function) -> str:
        fields = {
            "params": self.build_list(map(self.visit_expr, op.params)),
            "body": self.visit_expr(op.body),
            "ret_shape": self.visit_expr(op.ret_shape),
        }
        if op.ret_type:
            fields["ret_type"] = self.visit_type_(op.ret_type)
        if op.attrs:
            fields["attrs"] = self.build_list(
                map(
                    lambda kv: f"{wrap_quotes(str(kv[0]))}: {wrap_quotes(str(kv[1]))}",
                    op.attrs.items(),
                ),
                open_tok="{",
                close_tok="}",
            )
        return self.build_ast_node("Function", **fields)

    def visit_call_(self, op: relax.Call) -> str:
        fields = {
            "op": self.visit_expr(op.op),
            "args": self.build_list(map(self.visit_expr, op.args)),
        }
        if op.type_args:
            fields["type_args"] = self.build_list(map(self.visit_type_, op.type_args))
        if op.attrs and self.include_call_attrs:

            def display_attrs(attr_key):
                attr_val = op.attrs[attr_key]
                # attrs can be strings but also other types;
                # we want to wrap strings in quotes
                # (__repr__ would work but it uses single quotes)
                attr_str = wrap_quotes(attr_val) if isinstance(attr_val, str) else str(attr_val)
                return f"{wrap_quotes(attr_key)}: {attr_str}"

            fields["attrs"] = self.build_list(
                map(display_attrs, op.attrs.keys()),
                open_tok="{",
                close_tok="}",
            )
        return self.build_ast_node("Call", **fields)

    def visit_seq_expr_(self, op: relax.SeqExpr) -> str:
        return self.build_ast_node(
            "SeqExpr",
            blocks=self.build_list(map(self.visit_binding_block_, op.blocks)),
            body=self.visit_expr(op.body),
        )

    def visit_if_(self, op: relax.If) -> str:
        return self.build_ast_node(
            "If",
            cond=self.visit_expr(op.cond),
            true_branch=self.visit_expr(op.true_branch),
            false_branch=self.visit_expr(op.false_branch),
        )

    def visit_op_(self, op: tvm.ir.Op) -> str:
        # TODO: List other attributes?
        return self.build_ast_node("Op", name=wrap_quotes(op.name))

    def visit_prim_expr_(self, prim_expr: PrimExpr) -> str:
        # TODO: We may want to print PrimExpr ASTs, but this is a simplification for now
        return self.build_ast_node("PrimExpr", value=f"`{str(prim_expr)}`")

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> str:
        return self.build_ast_node(
            "TupleGetItem",
            tuple_value=self.visit_expr(op.tuple_value),
            index=str(op.index),
        )

    def visit_type_(self, type_node: relax.Type) -> str:
        """
        Recurse down types and print their ASTs too
        """
        if isinstance(type_node, relax.ShapeType):
            return self.build_ast_node("ShapeType")
        if isinstance(type_node, relax.ObjectType):
            return self.build_ast_node("ObjectType")
        if isinstance(type_node, relax.DynTensorType):
            fields = {}
            if type_node.ndim is not None:
                fields["ndim"] = str(type_node.ndim)
            if type_node.dtype != "":
                fields["dtype"] = type_node.dtype
            return self.build_ast_node("DynTensorType", **fields)
        if isinstance(type_node, relax.DimType):
            return self.build_ast_node("DimType")
        if isinstance(type_node, relax.TupleType):
            return self.build_ast_node(
                "TupleType", fields=self.build_list(map(self.visit_type_, type_node.fields))
            )
        if isinstance(type_node, relax.FuncType):
            return self.build_ast_node(
                "FuncType",
                arg_types=self.build_list(map(self.visit_type_, type_node.arg_types)),
                ret_type=self.visit_type_(type_node.ret_type),
                # TODO: skipping type params and type constraints
            )
        raise ValueError(f"Invalid Relax Type {type_node} ({type(type_node)})")

    def visit_binding_block_(self, block: relax.BindingBlock) -> str:
        """
        Recurse down binding blocks
        """
        return self.build_ast_node(
            "BindingBlock",
            bindings=self.build_list(map(self.visit_binding_, block.bindings), force_newline=True),
        )

    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> str:
        """
        Recurse down a dataflow block
        """
        return self.build_ast_node(
            "DataflowBlock",
            bindings=self.build_list(map(self.visit_binding_, block.bindings), force_newline=True),
        )

    def visit_binding_(self, binding: relax.Binding) -> str:
        """
        Distinguish between binding types
        """
        if isinstance(binding, relax.MatchShape):
            return self.visit_match_shape_(binding)
        if isinstance(binding, relax.VarBinding):
            return self.visit_var_binding_(binding)
        raise ValueError(f"Invalid binding type in {binding}: {type(binding)}")

    def visit_match_shape_(self, match_shape: relax.MatchShape) -> str:
        """
        Handle match shape
        """
        return self.build_ast_node(
            "MatchShape",
            value=self.visit_expr(match_shape.value),
            pattern=self.build_list(map(self.visit_prim_expr_, match_shape.pattern)),
            var=self.visit_expr(match_shape.var),
        )

    def visit_var_binding_(self, var_binding: relax.VarBinding) -> str:
        """
        Handle ordinary var bindings
        """
        return self.build_ast_node(
            "VarBinding",
            var=self.visit_expr(var_binding.var),
            value=self.visit_expr(var_binding.value),
        )


def dump_ast(
    exp: relax.Expr,
    indent_str="    ",
    include_type_annotations=True,
    include_shape_annotations=True,
    include_call_attrs=True,
) -> str:
    """
    Dump an AST in a text format.
    Can vary the indentation string and choose whether to include
    type and shape annotations or call attributes.
    """
    printer = ASTPrinter(
        indent_str=indent_str,
        include_type_annotations=include_type_annotations,
        include_shape_annotations=include_shape_annotations,
        include_call_attrs=include_call_attrs,
    )
    return printer.visit_expr(exp)
