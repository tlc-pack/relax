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
# pylint: disable=missing-docstring
"""AST Evaluation"""
import ast
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from . import dispatch, doc

if TYPE_CHECKING:
    from .parser import Parser

DEFAULT_OP: Dict[Type, Callable[..., Any]] = {
    doc.Add: lambda a, b: a + b,
    doc.Sub: lambda a, b: a - b,
    doc.Mult: lambda a, b: a * b,
    doc.Div: lambda a, b: a / b,
    doc.FloorDiv: lambda a, b: a // b,
    doc.Mod: lambda a, b: a % b,
    doc.LShift: lambda a, b: a << b,
    doc.RShift: lambda a, b: a >> b,
    doc.BitOr: lambda a, b: a | b,
    doc.BitXor: lambda a, b: a ^ b,
    doc.BitAnd: lambda a, b: a & b,
    doc.MatMult: lambda a, b: a @ b,
    # fmt: off
    doc.Pow: lambda a, b: a**b,
    # fmt: on
    doc.Eq: lambda a, b: a == b,
    doc.NotEq: lambda a, b: a != b,
    doc.Lt: lambda a, b: a < b,
    doc.LtE: lambda a, b: a <= b,
    doc.Gt: lambda a, b: a > b,
    doc.GtE: lambda a, b: a >= b,
    doc.Is: lambda a, b: a is b,
    doc.IsNot: lambda a, b: a is not b,
    doc.In: lambda a, b: a in b,
    doc.NotIn: lambda a, b: a not in b,
    doc.And: lambda a, b: a and b,
    doc.Or: lambda a, b: a or b,
    doc.Invert: lambda a: ~a,
    doc.Not: lambda a: not a,
    doc.UAdd: lambda a: +a,
    doc.USub: lambda a: -a,
}


class ExprEvaluator:

    parser: "Parser"
    value_table: Dict[str, Any]
    new_value_count: int

    def __init__(self, parser: "Parser", value_table: Dict[str, Any]) -> None:
        super().__init__()
        self.parser = parser
        self.value_table = value_table
        self.new_value_count = 0

    @staticmethod
    def eval(parser: "Parser", value_table: Dict[str, Any], node: doc.AST) -> Any:
        self = ExprEvaluator(parser, value_table)
        result = self._visit(node)  # pylint: disable=protected-access
        if isinstance(result, doc.Name):
            if result.id not in self.value_table:
                self.parser.report_error(result, f"Undefined variable: {result.id}")
            return self.value_table[result.id]
        if isinstance(result, doc.Constant):
            return result.value
        raise TypeError(f"Unexpected result type: {type(result)}")

    def _add_intermediate_result(self, value: Any) -> doc.Name:
        name = f"__tvm_tmp_value_{self.new_value_count}"
        self.new_value_count += 1
        self.value_table[name] = value
        lineno = 0
        col_offset = 0
        return doc.Name(
            id=name,
            ctx=doc.Load(
                lineno=lineno,
                col_offset=col_offset,
                end_lineno=None,
                end_col_offset=None,
            ),
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=None,
            end_col_offset=None,
        )

    def _visit(self, node: doc.AST) -> Any:
        if isinstance(node, list):
            return [self._visit(n) for n in node]
        if isinstance(node, tuple):
            return tuple(self._visit(n) for n in node)
        assert isinstance(node, doc.AST)
        if isinstance(node, doc.Name):
            if node.id not in self.value_table:
                self.parser.report_error(node, f"Undefined variable: {node.id}")
            return node
        if isinstance(
            node,
            (
                doc.Constant,
                doc.expr_context,
                doc.operator,
                doc.boolop,
                doc.unaryop,
                doc.cmpop,
            ),
        ):
            return node
        if not isinstance(node, (doc.expr, doc.slice)):
            return node
        if isinstance(node, doc.Lambda):
            return self._eval_lambda(node)
        fields = {}
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            attr = getattr(node, field)
            if isinstance(attr, (doc.AST, tuple, list)):
                fields[field] = self._visit(attr)
            else:
                fields[field] = attr
        try:
            if isinstance(node, doc.BoolOp):
                value = self._eval_bool_op(fields)
            elif isinstance(node, doc.Compare):
                value = self._eval_compare(fields)
            elif isinstance(node, doc.UnaryOp):
                value = self._eval_unary_op(fields)
            elif isinstance(node, doc.BinOp):
                value = self._eval_bin_op(fields)
            elif isinstance(node, doc.Slice):
                value = self._eval_slice(fields)
            else:
                value = self._eval_expr(node.__class__(**fields))
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.parser.report_error(node, str(e))
        return self._add_intermediate_result(value)

    def _eval_lambda(self, node: doc.Lambda) -> Any:
        try:
            value = self._eval_expr(node)
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.parser.report_error(node, str(e))
        return self._add_intermediate_result(value)

    def _eval_bool_op(self, fields: Dict[str, Any]) -> Any:
        op = fields["op"]
        if not isinstance(op, (doc.And, doc.Or)):
            raise TypeError(f"Unexpected operator: {op}")
        value = self._eval_expr(fields["values"][0])
        for rhs in fields["values"][1:]:
            value = _eval_op(op, values=[value, self._eval_expr(rhs)])
        return value

    def _eval_compare(self, fields: Dict[str, Any]) -> Any:
        value = self._eval_expr(fields["left"])
        for op, rhs in zip(fields["ops"], fields["comparators"]):
            value = _eval_op(op, values=[value, self._eval_expr(rhs)])
        return value

    def _eval_unary_op(self, fields: Dict[str, Any]) -> Any:
        value = self._eval_expr(fields["operand"])
        value = _eval_op(fields["op"], values=[value])
        return value

    def _eval_bin_op(self, fields: Dict[str, Any]) -> Any:
        return _eval_op(
            fields["op"],
            values=[
                self._eval_expr(fields["left"]),
                self._eval_expr(fields["right"]),
            ],
        )

    def _eval_slice(self, fields: Dict[str, Any]) -> Any:
        lower, upper, step = fields["lower"], fields["upper"], fields["step"]

        lower = self._eval_expr(lower) if lower is not None else None
        upper = self._eval_expr(upper) if upper is not None else None
        step = self._eval_expr(step) if step is not None else None

        return slice(lower, upper, step)

    def _eval_expr(self, v: Any) -> Any:
        return _eval_expr(v, self.value_table)


def eval_expr(
    parser: "Parser",
    node: Union[doc.expr, doc.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    value_table = {}
    if dict_globals is not None:
        value_table.update(dict_globals)
    return ExprEvaluator.eval(parser, value_table, node)


def eval_assign(
    parser: "Parser",
    target: doc.expr,
    source: Any,
) -> Dict[str, Any]:
    try:
        return _eval_assign(target, source)
    except Exception as e:  # pylint: disable=broad-except,invalid-name
        parser.report_error(target, f"Failed to evaluate assignment: {str(e)}")
        raise


def _eval_expr(
    node: Union[doc.expr, doc.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    node = doc.from_doc(node)
    if isinstance(node, ast.expr):
        node = ast.Expression(body=node)
    assert isinstance(node, ast.Expression), "Expects an ast.Expression, but gets: " + str(node)
    if dict_globals is None:
        dict_globals = {}
    node = ast.fix_missing_locations(node)
    exe = compile(node, filename="<ast>", mode="eval")
    return eval(exe, dict_globals)  # pylint: disable=eval-used


def _eval_op(
    op: doc.AST,
    values: List[Any],
):
    op_type = type(op)  # pylint: disable=protected-access
    for i, v in enumerate(values):
        v_type = getattr(type(v), "_dispatch_type", None)
        if v_type is None:
            continue
        f = dispatch.get_op(ty=v_type, op=op_type, operand_index=i, default=None)
        if f is not None:
            return f(*values)
    return DEFAULT_OP[op_type](*values)


def _eval_assign(
    target: doc.expr,
    source: Any,
) -> Dict[str, Any]:
    target = doc.from_doc(target)
    assert isinstance(target, ast.expr)
    RHS_VAR_NAME = "__tvm_rhs_var__"  # pylint: disable=invalid-name
    rhs_var_name = RHS_VAR_NAME
    dict_locals = {rhs_var_name: source}
    mod = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.Assign(
                    targets=[target],
                    value=ast.Name(
                        id=rhs_var_name,
                        ctx=ast.Load(),
                    ),
                )
            ],
            type_ignores=[],
        )
    )
    exe = compile(mod, filename="<ast>", mode="exec")
    exec(exe, {}, dict_locals)  # pylint: disable=exec-used
    del dict_locals[rhs_var_name]
    return dict_locals
