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
# pylint: disable=invalid-name, no-else-return, too-many-nested-blocks
# pylint: disable=inconsistent-return-statements, ungrouped-imports
"""TVM Script Parser For Relax"""
from __future__ import annotations

import inspect
import json
from enum import Enum
from typing import Union, Dict, List, Tuple, Optional, Callable, Any

import tvm
from tvm import relay, relax, tir
from tvm.relax.utils import metadata_partitioner
import tvm.script
from tvm.ir import diagnostics
from tvm.ir.module import IRModule
from tvm.script.tir.node import BufferSlice
import tvm.script.tir as tir_namespace
import tvm.script.relax as relax_namespace
import synr
from synr import ast, Transformer

from ..parser import TVMScriptParser as _TIRScriptParser
from ..utils import tvm_span_from_synr, call_with_error_reporting


def _is_registered(op_name: str, op_set=None) -> bool:
    """Returns whether or not the given operator is registered.

    Parameters
    ----------
    op_name : str
        The name of the operator.
    op_set : Union[Container, Iterable], optional
        The collection of registered operator names to check against. If None, the global TVM
        operator registry is queried.

    Returns
    -------
    bool
        True if the specified operator is registered, else False.
    """
    if op_set is None:
        op_set = tvm.ir._ffi_api.ListOpNames()
    return op_name in op_set


# NOTE: call_tir is an actual registered operator
class SpecialOp(Enum):
    """Relax and TIR operators that have special semantics handled by the parser."""

    MATCH_SHAPE = "relax.match_shape"
    CALL_PACKED = "relax.call_packed"
    DATAFLOW = "relax.dataflow"
    DATAFLOW_OUTPUT = "relax.output"
    TUPLE = "relax.Tuple"
    TUPLE_GET_ITEM = "relax.TupleGetItem"
    CONST = "relax.const"
    CONSTANT = "relax.expr.Constant"
    TIR_CAST = "tir.cast"
    TIR_MAX = "tir.max"


class ArithmeticOp(Enum):
    """Arithmetic operators that can desugar to either Relax or TIR PrimExpr operators."""

    ADD = ast.BuiltinOp.Add
    SUB = ast.BuiltinOp.Sub
    MUL = ast.BuiltinOp.Mul
    DIV = ast.BuiltinOp.Div
    FLOOR_DIV = ast.BuiltinOp.FloorDiv


RELAX_ARITHMETIC_OP_MAP = {
    ArithmeticOp.ADD: relay.op.get("add"),
    ArithmeticOp.SUB: relay.op.get("subtract"),
    ArithmeticOp.MUL: relay.op.get("multiply"),
    ArithmeticOp.DIV: relay.op.get("divide"),
    ArithmeticOp.FLOOR_DIV: relay.op.get("floor_divide"),
}

PRIMEXPR_ARITHMETIC_OP_MAP = {
    ArithmeticOp.ADD: tir.Add,
    ArithmeticOp.SUB: tir.Sub,
    ArithmeticOp.MUL: tir.Mul,
    ArithmeticOp.DIV: tir.Div,
    ArithmeticOp.FLOOR_DIV: tir.FloorDiv,
}


class RelaxTransformer(Transformer):
    """A visitor to handle transformations on the Relax AST"""

    meta_attr = None

    def __init__(self, ir_mod: IRModule, relax_prefix: List[str], tir_prefix: List[str]):
        super().__init__()
        self.mod = ir_mod
        self.relax_prefix = relax_prefix
        self.tir_prefix = tir_prefix
        self._scopes = [{}]  # str -> Var
        self._registered_ops = set(tvm.ir._ffi_api.ListOpNames())  # cached

    def to_tvm_span(self, span: ast.Span) -> tvm.ir.Span:
        """Helper method for converting synr span to TVM span.

        Parameters
        ----------
        span : ast.Span
            The synr span

        Returns
        -------
        tvm.ir.Span
            The corresponding TVM span
        """
        return self._diagnostic_context.to_tvm_span(span)

    def report_error(self, msg: str, span: ast.Span):
        """Helper method for emitting and immediately rendering an error.

        Parameters
        ----------
        msg : str
            The error message
        span : ast.Span
            The span to report the error at
        """
        self._diagnostic_context.emit("error", msg, self.to_tvm_span(span))
        self._diagnostic_context.render()

    def new_scope(self):
        """Helper method for creating a new scope context object

        Returns
        -------
        _Scope
            An internal scope context object used in a with block to create a new scope
        """

        class _Scope:
            def __init__(self, transformer: "RelaxTransformer"):
                self.transformer = transformer

            def __enter__(self):
                self.transformer._scopes.append(self.transformer._scopes[-1].copy())

            def __exit__(self, *exc):
                assert len(self.transformer._scopes) > 1, "cannot pop root scope"
                self.transformer._scopes.pop()

        return _Scope(self)

    @classmethod
    def update_meta(cls, metadata: str):
        """Update the metadata attributes.

        Parameters
        ----------
        metadata : str
            The metadata to be parsed.
        """

        cls.meta_attr = metadata

    @classmethod
    def get_meta(cls) -> str:
        """Return the metadata attribute.

        Returns
        -------
        str:
            The metadata attributes
        """
        return cls.meta_attr

    @property
    def scope(self):
        """Returns the current definition scope.

        Returns
        -------
        Dict[str, Union[relax.Var, tir.Var]]
            The scope of all currently defined variables (Relax and TIR).
        """
        return self._scopes[-1]

    def decl_var(
        self,
        name: str,
        type_annotation: Optional[relax.Type],
        shape: Optional[relax.Expr],
        span: ast.Span,
        is_dataflow: bool = False,
    ) -> relax.Var:
        """Introduces a variable with the given name and annotations to the current scope.

        Parameters
        ----------
        name : str
            The name of the variable
        type_annotation : Optional[relax.Type]
            The type annotation
        shape : Optional[relax.Expr]
            The shape annotation
        span : ast.Span
            The span where the variable is declared

        Returns
        -------
        Union[relax.Var, relax.DataflowVar]
            The declared variable
        """
        if name in self.scope:
            # TODO(@altanh): maybe emit an error at the declaration site and report it together
            self.report_error("variable has already been declared in the current scope", span)
        if is_dataflow:
            var = relax.DataflowVar(name, shape, type_annotation, self.to_tvm_span(span))
        else:
            var = relax.Var(name, shape, type_annotation, self.to_tvm_span(span))
        self.scope[name] = var
        return var

    def parse_tensor_kwargs_value(
        self, ty: ast.Type, span=None
    ) -> Union[str, None, bool, complex, float, int]:
        """Parse value of Tensor annotation keyword parameters in synr ast and return value in
        primitive type.

        Parameters
        ----------
        ty : ast.Ty
            The value of one of Tensor annotation keyword paramters

        Returns
        -------
        Union[str, None, bool, complex, float, int]
            Parsed value in primitive type
        """

        if isinstance(ty, ast.TypeConstant):
            return ty.value

        if isinstance(ty, ast.TypeCall):
            if ty.func_name == ast.BuiltinOp.UAdd:
                assert len(ty.params) == 1

                val = self.parse_tensor_kwargs_value(ty.params[0], span)
                if not isinstance(val, int):
                    self.report_error(f"expected int, but got {type(val)}", span)
                return val

            if ty.func_name == ast.BuiltinOp.USub:
                assert len(ty.params) == 1

                val = self.parse_tensor_kwargs_value(ty.params[0], span)
                if not isinstance(val, int):
                    self.report_error(f"expected int, but got {type(val)}", span)
                return 0 - val
            self.report_error(f"unsupported op: {ty.func_name}", ty.span)

        self.report_error(f"unexpected value of keyword argument {ty}", ty.span)

    def parse_tensor_kwargs(self, ty: ast.Type) -> dict[str, int]:
        """Parse keyword parameters of Tensor type annotation.

        Parameters
        ----------
        ty : ast.Ty
            The Tensor type annotation

        Returns
        -------
        dict[str, int]
            Parsed keyword parameters in dict[key, value] format
        """
        kwargs = {}
        for key, val in ty.keyword_params.items():
            assert isinstance(key, ast.TypeConstant) and isinstance(key.value, str)
            kwargs[key.value] = self.parse_tensor_kwargs_value(val, ty.span)

        # sanity check for Tensor type keyword arguments
        if len(kwargs) == 0:
            return kwargs
        if not (len(kwargs) == 1 and "ndim" in kwargs.keys()):
            self.report_error(
                f"expected one keyword argument 'ndim' but got {list(kwargs)}", ty.span
            )
        if not isinstance(kwargs["ndim"], int):
            self.report_error(
                f"expcted 'ndim' to be of type int, but got {type(kwargs['ndim'])}", ty.span
            )
        if kwargs["ndim"] < -1:
            self.report_error(f"ndim must be >= -1, but got {kwargs['ndim']}", ty.span)
        return kwargs

    def parse_dyn_tensor_type(
        self, ty: Union[ast.Type, ast.Call], bind_free_vars: bool
    ) -> Tuple[relax.Type, relax.Expr]:
        """
        Transforms the given synr tensor type annotation to a Relax DynTensorType
        Parameters
        ----------
        ty : ast.Type or ast.Call
            The synr type
        bind_free_vars : bool
            Whether or not the shape annotation can introduce new dimension variables

        Returns
        -------
        Tuple[relax.Type, relax.Expr]:
            The corresponding Relax type and shape expression
        """

        # TODO(@altanh): forgetting dtype like "Tensor((n, m))" ends up getting parsed as
        #                Tensor(n, m) which makes correct errors difficult here...
        if len(ty.params) != 2:
            self.report_error(
                "Tensor type annotations must have 2 positional fields (shape and dtype)"
                " and one optional keyword field ndim",
                ty.span,
            )

        shape_annotation, dtype_annotation = ty.params
        shape, dtype, ndim = None, None, -1

        # parse the shape annotation
        if isinstance(shape_annotation, ast.TypeConstant) and shape_annotation.value is None:
            pass  # shape = None
        elif isinstance(shape_annotation, ast.TypeVar):
            if shape_annotation.id.name != "_":
                # TODO(@altanh): handle variable annotations, e.g. x: Tensor(my_shape, _)
                self.report_error(
                    "variable Tensor shape annotations not yet supported",
                    shape_annotation.span,
                )
            else:
                shape = relax.RuntimeDepShape(span=self.to_tvm_span(shape_annotation.span))
        elif isinstance(shape_annotation, ast.TypeTuple):
            shape = relax.ShapeExpr(
                self.parse_shape(shape_annotation, bind_free_vars),
                span=self.to_tvm_span(shape_annotation.span),
            )
            ndim = len(shape)
        elif isinstance(shape_annotation, ast.Tuple):
            shape = relax.ShapeExpr(
                self.parse_shape(shape_annotation, bind_free_vars),
                span=self.to_tvm_span(shape_annotation.span),
            )
            ndim = len(shape)

        else:
            self.report_error(
                f"unsupported shape annotation {shape_annotation}",
                shape_annotation.span,
            )

        # parse the dtype annotation
        if isinstance(dtype_annotation, ast.TypeVar) and dtype_annotation.id.name == "_":
            pass  # dtype = None
        elif isinstance(dtype_annotation, ast.TypeConstant):
            dtype = dtype_annotation.value
        elif isinstance(dtype_annotation, ast.Constant):
            dtype = dtype_annotation.value
        else:
            self.report_error(
                "Tensor dtype annotations must be concrete or erased",
                dtype_annotation.span,
            )
        # parse optional keyword argument "ndim" if present
        kwargs = self.parse_tensor_kwargs(ty)
        if "ndim" in kwargs.keys():
            # If ndim was also inferred from shape annotation, then it must match keyword
            # argument ndim.
            if ndim >= 0 and kwargs["ndim"] != ndim:
                self.report_error(
                    f"#shape dimensions must match ndim: {ndim} vs. {kwargs['ndim']}",
                    ty.span,
                )
            else:
                ndim = kwargs["ndim"]
        span = self.to_tvm_span(ty.span)
        return (relax.DynTensorType(ndim=ndim, dtype=dtype, span=span), shape)

    def transform_type(self, ty: ast.Type, bind_free_vars: bool) -> Tuple[relax.Type, relax.Expr]:
        """Transforms the given synr type annotation to a Relax type and shape expression.

        Parameters
        ----------
        ty : ast.Type
            The synr type
        bind_free_vars : bool
            Whether or not the shape annotation can introduce new dimension variables

        Returns
        -------
        Tuple[relax.Type, relax.Expr]:
            The corresponding Relax type and shape expression
        """
        if ty is None:
            return (None, None)

        span = self.to_tvm_span(ty.span)

        # simple annotation with no type arguments
        if isinstance(ty, ast.TypeVar):
            if ty.id.name == "Tensor":
                return (relax.DynTensorType(ndim=-1, dtype=None, span=span), None)
            elif ty.id.name == "Shape":
                return (relax.ShapeType(span), None)
            elif ty.id.name == "Object":
                return (relax.ObjectType(span), None)
            elif ty.id.name == "Dim":
                return (relax.DimType(span), None)
            self.report_error("unknown type in annotation", ty.span)

        # annotation with type arguments/shape annotation
        if isinstance(ty, ast.TypeCall):
            if ty.func_name.id.name == "Tensor":
                return self.parse_dyn_tensor_type(ty, bind_free_vars)
            elif ty.func_name.id.name == "Tuple":
                field_types = []
                field_shapes = []
                for field in ty.params:
                    fty, fsh = self.transform_type(field, bind_free_vars=False)
                    field_types.append(fty)
                    field_shapes.append(fsh)
                return (relay.TupleType(field_types, span), None)
            elif ty.func_name.id.name == "Callable":
                if len(ty.params) != 2:
                    self.report_error(
                        "Function type annotations must have 2 positional fields",
                        ty.span,
                    )

                func_arg_types, func_ret_type = ty.params
                input_tensors = []
                # Single input
                if isinstance(func_arg_types, ast.TypeCall):
                    tensor_type = self.parse_dyn_tensor_type(func_arg_types, bind_free_vars)
                    input_tensors.append(tensor_type[0])
                # Multiple inputs
                elif isinstance(func_arg_types, ast.TypeTuple):
                    for func_arg_type in func_arg_types.values:
                        tensor_type = self.parse_dyn_tensor_type(func_arg_type, bind_free_vars)
                        input_tensors.append(tensor_type[0])
                else:
                    self.report_error(
                        "Function Reture Type annotations must be concrete or erased",
                        func_arg_types.span,
                    )

                ret_type = self.transform_type(func_ret_type, bind_free_vars)

                return (relax.FuncType(input_tensors, ret_type[0]), None)

        self.report_error("invalid type", ty.span)

    def parse_shape(
        self,
        shape_annotation: Union[ast.TypeTuple, ast.Tuple],
        bind_free_vars: bool,
    ) -> List[tir.PrimExpr]:
        """Parses the given shape annotation to a list of PrimExprs.

        Parameters
        ----------
        shape_annotation : Union[ast.TypeTuple, ast.Tuple]
            The shape annotation in synr
        bind_free_vars : bool
            Whether or not the annotation can bind previously free variables

        Returns
        -------
        List[tir.PrimExpr]
            The parsed shape as a list of PrimExprs
        """
        return [self.parse_primexpr(field, bind_free_vars) for field in shape_annotation.values]

    def parse_primexpr(self, expr: ast.Expr, bind_free_vars: bool) -> tir.PrimExpr:
        """Parses the given expression to a PrimExpr.

        Parameters
        ----------
        expr : ast.Expr
            The input expression
        bind_free_vars : bool
            Whether or not the expression can bind previously free variables

        Returns
        -------
        tir.PrimExpr
            The result PrimExpr
        """
        if isinstance(expr, ast.Var):
            var_name = expr.id.name
            if var_name in self.scope:
                var = self.scope[var_name]
                if not isinstance(var, tir.Var):
                    # TODO(@altanh): we may wish to relax this in the future to support constructing
                    #                shapes from Dim-typed Relax expressions
                    self.report_error(
                        "non-dimension variables cannot appear in dimension expressions",
                        expr.span,
                    )
                return var
            elif bind_free_vars:
                # introduce TIR variable to scope, e.g. for func params or relax.call_packed
                var = tir.Var(var_name, "int64", self.to_tvm_span(expr.span))
                self.scope[var_name] = var
                return var
            else:
                self.report_error(
                    "cannot introduce new dimension variables in this expression",
                    expr.span,
                )

        elif isinstance(expr, ast.Constant):
            if not isinstance(expr.value, int):
                self.report_error("only integer constants are supported", expr.span)
            return tir.const(expr.value, "int64", self.to_tvm_span(expr.span))

        elif isinstance(expr, ast.Call):
            if not isinstance(expr.func_name, ast.Op):
                self.report_error(
                    "only built-in operators can be used in dimension expressions",
                    expr.func_name.span,
                )
            op = PRIMEXPR_ARITHMETIC_OP_MAP[self.transform_expr(expr.func_name)]
            # TODO(@altanh): it might not make sense to bind free variables
            args = [self.parse_primexpr(arg, bind_free_vars) for arg in expr.params]
            return op(*args, span=self.to_tvm_span(expr.span))

        else:
            self.report_error(f"unsupported dimension expression: {expr}", expr.span)

    def transform_module(self, mod: ast.Module) -> IRModule:
        """Transforms the given synr Module to a Relax IRModule or Function.

        Parameters
        ----------
        mod : ast.Module
            The input synr Module

        Returns
        -------
        Union[IRModule, Function]
            The parsed Relax IRModule or Function
        """
        if len(mod.funcs) != 1:
            self.report_error(
                "the input must be either a single function or a single class", mod.span
            )

        (root_func,) = mod.funcs.values()

        if isinstance(root_func, ast.Function):
            return self.transform_function(root_func, is_global=True)
        elif isinstance(root_func, ast.Class):
            # add global vars to the root scope for resolving global function calls
            for func_name in root_func.funcs:
                self.scope[func_name] = relay.GlobalVar(func_name)
            for func_name, func in root_func.funcs.items():
                global_var = self.scope[func_name]
                self.mod[global_var] = self.transform_function(func, is_global=True)

            # TODO(@yuchen): temporarily make the the parser.from_source api also run
            # ResolveGlobals pass to populate shape and checked type to be consitent
            # with the behavior of directly parsing TVMScript
            self.mod = relax.transform.Normalize()(self.mod)
            self.mod = relax.transform.ResolveGlobals()(self.mod)
            return self.mod
        else:
            self.report_error(f"unsupported input class: {root_func}", root_func.span)

    def _parse_attrs_to_str(self, expr: ast.Attr) -> str:
        strs = []
        attr = expr
        while isinstance(attr, ast.Attr):
            strs.append(attr.field.name)
            attr = attr.object
        if not isinstance(attr, ast.Var):
            self.report_error("unsupported attribute access", expr.span)
        if attr.id.name in self.tir_prefix:
            strs.append("tir")
        elif attr.id.name in self.relax_prefix:
            strs.append("relax")
        else:
            strs.append(attr.id.name)
        result = ".".join(reversed(strs))
        return result

    def _get_lhs(self, stmt: ast.Assign) -> ast.Var:
        if len(stmt.lhs) > 1:
            self.report_error("currently we only support single variable assignments", stmt.span)
        return stmt.lhs[0]

    def _tir_from_synr(self, synr_ast: ast.Node) -> tir.PrimFunc:
        """Parses the given synr AST using the TVMScript parser to a PrimFunc.

        Parameters
        ----------
        synr_ast : ast.Node
            The synr AST to be parsed.
        diag_ctx : _TIRDiagnosticCtx
            The diagnostic context for TVMScript parser error reporting.

        Returns
        -------
        tir.PrimFunc
            The parsed TIR PrimFunc.
        """
        # this behavior is assumed by the TIR parser
        self._diagnostic_context._render_on_error = True
        parser = _TIRScriptParser(synr_ast.span.start_line, self.tir_prefix, {})
        prim_func = parser.do_transform(synr_ast, self._diagnostic_context)
        self._diagnostic_context._render_on_error = False
        return prim_func

    def transform_function(self, func: ast.Function, is_global: bool = False) -> relax.Function:
        """Transforms the given synr Function to a Relax Function.

        Parameters
        ----------
        func : ast.Function
            The input synr Function
        is_global : bool, optional
            Whether or not the input function is global/module-level, by default False

        Returns
        -------
        relax.Function
            The parsed Relax Function
        """
        if len(func.decorators) != 1:
            self.report_error(
                "functions must be decorated as a Relax Function or TIR PrimFunc", func.span
            )
        decorator_name = None
        if isinstance(func.decorators[0], ast.Call):
            decorator_name = self._parse_attrs_to_str(func.decorators[0].func_name)
        else:
            decorator_name = self._parse_attrs_to_str(func.decorators[0])

        if decorator_name == "tir.prim_func":
            return self._tir_from_synr(func)

        if decorator_name != "relax.function":
            self.report_error(
                "functions must be decorated as a Relax Function or TIR PrimFunc", func.span
            )

        with self.new_scope():
            params = []
            for param in func.params:
                ty, shape = self.transform_type(param.ty, bind_free_vars=True)
                param = self.decl_var(param.name, ty, shape, param.span)
                params.append(param)
            new_body = self.transform_block(func.body)
            ret_type, _ = self.transform_type(func.ret_type, bind_free_vars=False)

        relax_func = relax.Function.create_unchecked(
            params,
            new_body,
            ret_type,
            attrs=None,
            span=self.to_tvm_span(func.span),
        )
        if is_global:
            relax_func = relax_func.with_attr("global_symbol", func.name)

        return relax_func

    def is_match_shape(self, stmt: ast.Stmt) -> bool:
        """Returns whether or not the given statement is a MatchShape binding.

        Parameters
        ----------
        stmt : ast.Stmt
            The statement to be parsed.

        Returns
        -------
        bool
            Whether or not the statement is a MatchShape binding.
        """
        call_op = None
        if isinstance(stmt, ast.UnassignedCall):
            call_op = self.transform_expr(stmt.call.func_name)
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.rhs, ast.Call):
            call_op = self.transform_expr(stmt.rhs)
        return call_op == SpecialOp.MATCH_SHAPE

    def parse_binding(self, stmt: ast.Stmt, is_dataflow: bool = False) -> relax.Binding:
        """Parses the input synr statement to the corresponding Relax binding.

        Parameters
        ----------
        stmt : ast.Stmt
            The input synr statement (either an assignment or a unassigned call)
        is_dataflow : bool, optional
            Whether or not the binding is in a dataflow block, by default False

        Returns
        -------
        relax.Binding
            The parsed Relax binding
        """
        assert isinstance(stmt, (ast.Assign, ast.UnassignedCall))
        if self.is_match_shape(stmt):
            return self.parse_shape_binding(stmt, is_dataflow=is_dataflow)
        else:
            assert isinstance(stmt, ast.Assign)
            return self.parse_var_binding(stmt, is_dataflow=is_dataflow)

    def parse_shape_binding(self, stmt: ast.Stmt, is_dataflow: bool = False) -> relax.MatchShape:
        """Parses the input synr statement to a Relax shape binding.

        Parameters
        ----------
        stmt : ast.Stmt
            The input synr statement
        is_dataflow : bool, optional
            Whether or not the bound variable (if any) is a dataflow variable, by default False

        Returns
        -------
        relax.MatchShape
            The parsed Relax shape binding
        """
        var: ast.Var = None
        call: ast.Call = None

        if isinstance(stmt, ast.UnassignedCall):
            # case where only dimension variables are bound, e.g. `match_shape(x.shape, (n, m))`
            call = stmt.call
        else:
            # case where the statement also binds a Relax variable to the value being matched
            assert isinstance(stmt, ast.Assign)
            var = self._get_lhs(stmt)
            call = stmt.rhs
            if not isinstance(var, ast.Var):
                self.report_error("the left hand side of a binding must be a variable", stmt.span)

        op = self.transform_expr(call.func_name)

        assert op == SpecialOp.MATCH_SHAPE
        if len(call.params) != 2:
            self.report_error(op.value + " takes exactly two arguments", call.span)

        value, pattern = call.params

        value = self.transform_expr(value)
        if not isinstance(pattern, ast.Tuple):
            self.report_error(f"the pattern of a {op.value} call must be a tuple", pattern.span)
        pattern = self.parse_shape(pattern, bind_free_vars=True)

        if var is not None:
            # TODO(@altanh): keep or discard annotation?
            ty, shape = self.transform_type(stmt.ty, bind_free_vars=False)
            var = self.decl_var(var.id.name, ty, shape, var.span, is_dataflow=is_dataflow)

        return relax.MatchShape(value, pattern, var, self.to_tvm_span(stmt.span))

    def parse_var_binding(self, stmt: ast.Assign, is_dataflow=False) -> relax.VarBinding:
        """Parses the input synr assignment to a Relax variable binding.

        Parameters
        ----------
        stmt : ast.Assign
            The input synr assignment
        is_dataflow : bool, optional
            Whether or not the bound variable is a dataflow variable, by default False

        Returns
        -------
        relax.VarBinding
            The parsed Relax variable binding
        """
        var = self._get_lhs(stmt)
        if isinstance(stmt.rhs, ast.Constant):
            rhs = relax.const(stmt.rhs.value)
        else:
            rhs = self.transform_expr(stmt.rhs)
        # an ExternFunc call comes from call_packed
        bind_free_vars = isinstance(rhs, relay.Call) and isinstance(rhs.op, relax.ExternFunc)
        ty, shape = self.transform_type(stmt.ty, bind_free_vars)
        lhs = self.decl_var(var.id.name, ty, shape, var.span, is_dataflow=is_dataflow)
        return relax.VarBinding(lhs, rhs, self.to_tvm_span(stmt.span))

    # Stmts:
    # - Assert: probably unsupported for now
    # - Assign: VarBinding
    # - For: ??
    # - If: IfThenElse, must check no empty false branch
    # - Return: just the returned expression, must terminate blocks? (special case if-else)
    # - UnassignedCall: match_shape
    # - With: relax.dataflow
    def transform_stmt(
        self, stmt: ast.Stmt
    ) -> Union[relax.Expr, relax.Binding, relax.DataflowBlock]:
        """Transforms the given synr statement to the corresponding Relax node.

        Parameters
        ----------
        stmt : ast.Stmt
            The input synr statement

        Returns
        -------
        Union[relax.Expr, relax.Binding, relax.DataflowBlock]
            The parsed Relax node
        """
        if isinstance(stmt, ast.Assign):
            # dataflow bindings are handled separately in parse_dataflow
            return self.parse_binding(stmt)
        elif isinstance(stmt, ast.If):
            # check branches are non-empty
            if len(stmt.true.stmts) == 0 or len(stmt.false.stmts) == 0:
                self.report_error("both branches of an if-else block must be non-empty", stmt.span)
            true_assign = stmt.true.stmts[-1]
            false_assign = stmt.false.stmts[-1]

            # check last statement in each branch lines up
            if not isinstance(true_assign, ast.Assign) or not isinstance(
                self._get_lhs(true_assign), ast.Var
            ):
                self.report_error(
                    "each branch of an if-else statement must end in a variable assignment",
                    true_assign.span,
                )
            if not isinstance(false_assign, ast.Assign) or not isinstance(
                self._get_lhs(false_assign), ast.Var
            ):
                self.report_error(
                    "each branch of an if-else statement must end in a variable assignment",
                    false_assign.span,
                )
            union_span = ast.Span.union([true_assign.span, false_assign.span])
            if self._get_lhs(true_assign).id.name != self._get_lhs(false_assign).id.name:
                self.report_error(
                    "the final assignment of both branches must have the same variable",
                    union_span,
                )

            var_name = self._get_lhs(true_assign).id.name

            # rewrite branches to have a return statement so the blocks properly parse to SeqExprs
            true_block = synr.ast.Block(
                span=stmt.true.span,
                stmts=stmt.true.stmts[:-1]
                + [synr.ast.Return(span=true_assign.span, value=true_assign.rhs)],
            )
            false_block = synr.ast.Block(
                span=stmt.false.span,
                stmts=stmt.false.stmts[:-1]
                + [synr.ast.Return(span=false_assign.span, value=false_assign.rhs)],
            )

            # parse the branches, build the final expression and binding
            cond = self.transform_expr(stmt.condition)
            with self.new_scope():
                true_branch = self.transform_block(true_block)
            with self.new_scope():
                false_branch = self.transform_block(false_block)
            # TODO(@altanh): the spans here are all sorts of messed up, not sure how to fix
            ite_expr = relay.If(cond, true_branch, false_branch, self.to_tvm_span(stmt.span))
            # TODO(@altanh): type and shape of return var
            var = self.decl_var(var_name, None, None, union_span)
            return relax.VarBinding(var, ite_expr, self.to_tvm_span(union_span))

        elif isinstance(stmt, ast.Return):
            return self.transform_expr(stmt.value)

        elif isinstance(stmt, ast.UnassignedCall):
            if self.transform_expr(stmt.call.func_name) == SpecialOp.MATCH_SHAPE:
                return self.parse_shape_binding(stmt)
            else:
                self.report_error("the results of normal function calls must be bound", stmt.span)

        elif isinstance(stmt, ast.With):
            if not isinstance(stmt.rhs, ast.Call):
                self.report_error("unsupported with block", stmt.span)

            call = stmt.rhs
            op = self.transform_expr(call.func_name)

            # TODO(@altanh): perhaps this ought to be more general

            if op == SpecialOp.DATAFLOW:
                if len(call.params) > 0:
                    self.report_error(
                        "dataflow block constructor takes no arguments",
                        call.params[0].span,
                    )
                if len(stmt.lhs) > 0:
                    self.report_error(
                        "dataflow blocks don't bind any patterns",
                        stmt.lhs[0].span,
                    )
                return self.parse_dataflow(stmt.body)
            else:
                self.report_error("unsupported with block type", call.span)

        elif isinstance(stmt, ast.Function):
            func = self.transform_function(stmt)
            return func

        else:
            self.report_error(
                "unsupported statement",
                stmt.span,
            )

    def parse_dataflow(self, block: ast.Block) -> relax.DataflowBlock:
        """Parses the input synr block to a Relax dataflow block.

        Parameters
        ----------
        block : ast.Block
            The input synr block

        Returns
        -------
        relax.DataflowBlock
            The parsed Relax dataflow block
        """
        assert len(block.stmts) > 0, "should never have an empty dataflow block"
        bindings = []

        with self.new_scope():
            # parse the output statement first to figure out which bindings assign normal Vars
            output_stmt = block.stmts[-1]
            output_var_names = set()
            unbound_output_vars = {}
            output_vars = []

            if (
                isinstance(output_stmt, ast.UnassignedCall)
                and self.transform_expr(output_stmt.call.func_name) == SpecialOp.DATAFLOW_OUTPUT
            ):
                for var in output_stmt.call.params:
                    if not isinstance(var, ast.Var):
                        self.report_error(f"dataflow block outputs must be variables", var.span)
                    output_var_names.add(var.id.name)
                    unbound_output_vars[var.id.name] = var
            else:
                self.report_error(
                    f"dataflow blocks must end with a {SpecialOp.DATAFLOW_OUTPUT.value} statement",
                    output_stmt.span,
                )

            # output variables are bound to normal (not dataflow) Vars
            for binding_stmt in block.stmts[:-1]:
                if not isinstance(binding_stmt, (ast.Assign, ast.UnassignedCall)):
                    self.report_error(
                        "only bindings are supported in dataflow blocks",
                        binding_stmt.span,
                    )
                is_match_shape = self.is_match_shape(binding_stmt)
                is_dataflow = (
                    isinstance(binding_stmt, ast.Assign)
                    and self._get_lhs(binding_stmt).id.name not in output_var_names
                )
                binding = self.parse_binding(binding_stmt, is_dataflow=is_dataflow)
                bindings.append(binding)
                if not is_dataflow:
                    if is_match_shape:
                        for var in binding.pattern:
                            output_vars.append(var)
                    if binding.var is not None:
                        output_vars.append(binding.var)
                        unbound_output_vars.pop(binding.var.name_hint)

        # check that the output variables are all bound locally
        for unbound_var in unbound_output_vars.values():
            self._diagnostic_context.emit(
                "error",
                "dataflow output variables must be bound locally in the block",
                unbound_var.span,
            )
        # FIXME(@altanh): TVMDiagnosticCtx has hard-coded `emit` to always be an error and raise
        #                 an exception on the first call
        self._diagnostic_context.render()

        # make output variables visible in parent scope
        for v in output_vars:
            # v could already be in scope if it was a previously bound dimension variable
            v_name = v.name if isinstance(v, tir.Var) else v.name_hint
            if v not in self.scope:
                self.scope[v_name] = v

        return relax.DataflowBlock(bindings, self.to_tvm_span(block.span))

    def parse_attr(self, expr: ast.Attr) -> relax.Expr:
        """Parses the given synr Attr node to a Relax expression.

        Parameters
        ----------
        expr : ast.Attr
            The synr Attr node to be parsed.

        Returns
        -------
        relax.Expr
            The parsed expression.
        """
        if expr.field.name == "shape":
            obj = self.transform_expr(expr.object)
            return relay.Call(
                relay.op.get("relax.shape_of"), [obj], span=self.to_tvm_span(expr.span)
            )
        else:
            # assume it's a hierarchical op identifier (e.g. nn.softmax, relax.call_tir)
            op_name = self._parse_attrs_to_str(expr)
            # NOTE: at least for now, all special operators are namespaced
            try:
                return SpecialOp(op_name)
            except ValueError:
                # TODO(@altanh): maybe diagnostics here in case this fails?
                return relay.op.get(op_name)

    def parse_array_literal(
        self, expr: ast.ArrayLiteral
    ) -> Union[relax.const, relax.expr.Constant]:
        """Parses the given synr ArrayLiteral node to a Relax constant.

        Parameters
        ----------
        expr : ast.ArrayLiteral
            The synr ArrayLiteral to be parsed.

        Returns
        -------
        Union[relax.const, relax.expr.Constant]
            The parsed relex expression.
        """

        def _get_values(expr: ast.ArrayLiteral, vals: List[Any]) -> List[Any]:
            # todo(@yongwww): the generic parsing util for ArrayLiteral should be in synr
            if isinstance(expr, ast.Constant):
                vals.append(expr.value)
            elif isinstance(expr, ast.ArrayLiteral):
                for elem in expr.values:
                    # recursive call to get the nested list
                    nested_vals = _get_values(elem, [])
                    # avoid nested list for every element
                    if len(nested_vals) == 1 and not isinstance(nested_vals[0], list):
                        vals.append(nested_vals[0])
                    else:
                        vals.append(nested_vals)
            else:
                self.report_error(f"unsupported ast expression {expr}", expr.span)
            return vals

        const_values = _get_values(expr, [])
        return relax.const(const_values)

    # TODO(@tvm-team): Currenly the synr is over-specialized, unify with transform_type
    # to parse types in the future
    def parse_type_from_value(self, val: ast.Expr) -> relax.Type:
        """Parses the type_args value of a call to a Relax type.

        Parameters
        ----------
        val : ast.Expr
            The type_args value to be parsed.

        Returns
        -------
        relax.Type
            The parsed Relax type.
        """
        if isinstance(val, ast.Var):
            if val.id.name == "Tensor":
                return relax.DynTensorType(ndim=-1, dtype=None, span=self.to_tvm_span(val.span))
            elif val.id.name == "Object":
                return relax.ObjectType(self.to_tvm_span(val.span))
            elif val.id.name == "Shape":
                return relax.ShapeType(self.to_tvm_span(val.span))
            elif val.id.name == "Void":
                return relay.TupleType(None, self.to_tvm_span(val.span))
            else:
                self.report_error(
                    f"type_args value must be Tensor, Object, Shape, Void, or Tuple()", val.span
                )
        elif isinstance(val, ast.Call):
            if val.func_name.id.name == "Tensor":
                ndim = -1
                dtype = None
                for k, v in val.keyword_params.items():
                    if k.value == "ndim":
                        ndim = v.value
                    if k.value == "dtype":
                        dtype = v.value
                return relax.DynTensorType(ndim, dtype, self.to_tvm_span(val.span))
            elif val.func_name.id.name == "Tuple":
                field_types = []
                for field in val.params:
                    fty = self.parse_type_from_value(field)
                    field_types.append(fty)
                return relax.TupleType(field_types, self.to_tvm_span(val.span))
            else:
                self.report_error(
                    f"""type_args elements must be Tensor or Tuple when having arguments,
                    but meet {val.func_name.id.name}""",
                    val.span,
                )
        else:
            self.report_error(
                f"cannot parse {val} as the type_args value",
                val.span,
            )

    def parse_call_attr(self, expr: ast.Call) -> Tuple(tvm.ir.Attrs, List[relax.Type]):
        """Parses keyword parameters as call attributes.

        Parameters
        ----------
        expr : ast.Call
            The synr Call to be parsed.

        Returns
        -------
        Tuple(tvm.ir.Attrs, List[relax.Type])
            The parsed call attributes and type_args.
        """
        op = self.transform_expr(expr.func_name)
        kwargs = {}
        type_args = None
        for key, val in expr.keyword_params.items():
            if key.value == "type_args":
                type_args = self.parse_type_from_value(val)
                if type_args:
                    type_args = [type_args]
            else:
                assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                # TODO(@altanh): might need separate attribute parsing eventually
                kwargs[key.value] = self.transform_expr(val)

        is_default = False
        if "attrs_type_key" in kwargs:
            attrs_type_key = kwargs["attrs_type_key"]
            kwargs.pop("attrs_type_key")
        elif isinstance(op, tvm.ir.Op) and op.attrs_type_key != "":
            attrs_type_key = op.attrs_type_key
        else:
            attrs_type_key = "DictAttrs"
            is_default = True

        attrs = None
        if kwargs or not is_default:
            attrs = tvm.ir.attrs.make_node(attrs_type_key, **kwargs)
        return (attrs, type_args)

    def parse_call(self, expr: ast.Call) -> Union[tir.PrimExpr, relax.Expr]:
        """Parses the given synr Call node to a Relax expression or PrimExpr.

        Parameters
        ----------
        expr : ast.Call
            The synr Call node to be parsed.

        Returns
        -------
        Union[tir.PrimExpr, relax.Expr]
            The parsed expression. It will be a PrimExpr if expr is an arithmetic operation on
            PrimExprs.
        """
        if isinstance(expr.func_name, ast.Op) and expr.func_name.name == ast.BuiltinOp.Subscript:
            if (
                hasattr(expr.params[0], "params")
                and hasattr(expr.params[0].params[0], "id")
                and expr.params[0].params[0].id.name == "meta"
            ):
                # Get the index of constant in b64ndarrays in metadata
                const_idx = 0
                if hasattr(expr.params[-1], "values"):
                    const_idx = expr.params[-1].values[0].value

                if self.mod.get_attrs():
                    metadata = self.mod.get_attrs()
                else:
                    metadata = RelaxTransformer.get_meta()

                if not metadata:
                    self.report_error(
                        f"metadata is not found, please feed it into ir_module", expr.span
                    )

                attr_json = json.loads(str(metadata))
                new_root = const_num = 0
                for i, node in enumerate(attr_json["nodes"]):
                    if "type_key" in node and "Constant" in node["type_key"]:
                        if const_num == const_idx:
                            new_root = i
                            break
                        const_num += 1
                attr_json["root"] = new_root
                return tvm.ir.load_json(json.dumps(attr_json))
            else:
                return self.transform_Subscript(expr)

        op = self.transform_expr(expr.func_name)
        type_args = None

        if op == SpecialOp.CALL_PACKED:
            extern_func = expr.params[0]
            if not (isinstance(extern_func, ast.Constant) and isinstance(extern_func.value, str)):
                self.report_error(
                    "the first argument of " + op.value + " must be the extern function name",
                    extern_func.span,
                )
            op = relax.ExternFunc(extern_func.value, self.to_tvm_span(extern_func.span))
            args = [self.transform_expr(arg) for arg in expr.params[1:]]

        elif op == SpecialOp.TUPLE:
            args = [self.transform_expr(arg) for arg in expr.params[0].values]
            return relax.Tuple(args)

        elif op == SpecialOp.TUPLE_GET_ITEM:
            assert len(expr.params) == 2, "TupleGetItem expects to get two parameters."
            args = [self.transform_expr(arg) for arg in expr.params]
            # index of TupleGetItem only accepts int type intead of tir.expr.IntImm
            return relax.TupleGetItem(args[0], args[1].value)

        elif op in (SpecialOp.CONSTANT, SpecialOp.CONST):
            # relax const/Constant
            arg = expr.params[0]
            if isinstance(arg, ast.Constant):
                return relax.const(arg.value)
            elif isinstance(arg, ast.ArrayLiteral):
                return self.parse_array_literal(arg)
            else:
                self.report_error(f"unsupported ast for const: {arg}", expr.span)

        elif op == SpecialOp.TIR_CAST:
            if len(expr.params) != 2:
                self.report_error(
                    f"tir.cast expects 2 arguments, but got {len(expr.params)}", expr.span
                )
            args = [self.transform_expr(arg) for arg in expr.params]
            return tir.Cast(args[0], args[1])

        elif op == SpecialOp.TIR_MAX:
            if len(expr.params) != 2:
                self.report_error(
                    f"tir.max expects 2 arguments, but got {len(expr.params)}", expr.span
                )
            args = [self.transform_expr(arg) for arg in expr.params]
            return tir.Max(args[0], args[1])

        elif isinstance(op, ArithmeticOp):
            args = [self.transform_expr(arg) for arg in expr.params]
            if all([isinstance(arg, tir.PrimExpr) for arg in args]):
                return PRIMEXPR_ARITHMETIC_OP_MAP[op](*args, span=self.to_tvm_span(expr.span))
            # otherwise it's just a normal Relax operator call
            op = RELAX_ARITHMETIC_OP_MAP[op]

        elif isinstance(op, tvm.ir.Op):
            args = [self.transform_expr(arg) for arg in expr.params]
            # check call arity eagerly
            if op.name == "relax.call_tir":
                # call_tir is special case because last argument is optional
                if len(args) != op.num_inputs and len(args) != op.num_inputs - 1:
                    self.report_error(
                        f"""{op.name} expects {op.num_inputs} or {op.num_inputs - 1}
                        arguments but got {len(args)}""",
                        expr.span,
                    )

                if len(expr.keyword_params) != 1:
                    self.report_error(
                        f"""{op.name} expects exact one keyword argument with dtype as the key but
                        got {len(expr.keyword_params)} keyword arguments""",
                        expr.span,
                    )

                if isinstance(args[0], str):
                    # extern function call case: rewrite identifier to an ExternFunc
                    args[0] = relax.ExternFunc(args[0], self.to_tvm_span(expr.params[1].span))

                for key, val in expr.keyword_params.items():
                    assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                    if key.value == "dtype":
                        val = self.transform_expr(val)
                        # single output case
                        if isinstance(val, str):
                            if not isinstance(args[2], relax.ShapeExpr):
                                self.report_error(
                                    (
                                        f"The number of output_shape and output_dtype of "
                                        f"call_tir mismatch"
                                    ),
                                    expr.span,
                                )
                            type_args = [relax.DynTensorType(ndim=len(args[2].values), dtype=val)]
                        elif isinstance(val, Tuple):
                            # multiple outputs case
                            if not isinstance(args[2], Tuple) and len(args[2]) != len(val):
                                self.report_error(
                                    (
                                        f"The number of output_shape and output_dtype of "
                                        f"call_tir mismatch"
                                    ),
                                    expr.span,
                                )
                            types = []
                            for i in range(len(args[2])):
                                types.append(
                                    relax.DynTensorType(ndim=len(args[2][i].values), dtype=val[i])
                                )
                            type_args = [relax.TupleType(types)]
                        else:
                            self.report_error(
                                f"call_tir expects the output_dtype to be a string or a tuple",
                                expr.span,
                            )
                    else:
                        self.report_error(
                            (
                                f"{op.name} expects one keyword argument with dtype as the key but "
                                f"got {len(key.value)} as the key"
                            ),
                            expr.span,
                        )

            elif op.num_inputs != -1 and len(args) != op.num_inputs:
                self.report_error(
                    f"{op.name} expects {op.num_inputs} arguments but got {len(args)}", expr.span
                )

        elif isinstance(op, relay.Expr):
            args = [self.transform_expr(arg) for arg in expr.params]

        else:
            self.report_error(f"unsupported function in call: {op}", expr.func_name.span)

        if isinstance(op, tvm.ir.Op) and op.name == "relax.call_tir":
            attrs = None
        else:
            attrs, type_args = self.parse_call_attr(expr)

        if isinstance(op, relax.ExternFunc) and type_args is None:
            self.report_error(f"call_packed is required to have type_args", expr.span)

        return relax.Call(
            op, args, attrs=attrs, type_args=type_args, span=self.to_tvm_span(expr.span)
        )

    # Exprs:
    # - ArrayLiteral
    # - Attr: use for .shape, and intrinsic/special operator namespace
    # - Call
    # - Constant
    # - DictLiteral: unsupported for now
    # - Slice: unsupported for now, could desugar to slice op
    # - Tuple
    # - Var
    def transform_expr(self, expr: ast.Expr) -> relax.Expr:
        """Transforms the input synr expression to a Relax expression.

        Parameters
        ----------
        expr : ast.Expr
            The input synr

        Returns
        -------
        relax.Expr
            The corresponding Relax expression
        """

        if isinstance(expr, ast.Attr):
            return self.parse_attr(expr)

        elif isinstance(expr, ast.Call):
            if hasattr(expr.func_name, "field") and expr.func_name.field.name == "match_shape":
                return self.transform_expr(expr.func_name)
            return self.parse_call(expr)

        elif isinstance(expr, ast.Tuple):
            fields = [self.transform_expr(field) for field in expr.values]

            if all([isinstance(f, str) for f in fields]):
                return tuple(fields)

            # TODO(@altanh): this check might be too weak; we really only accept integral PrimExprs
            #                (e.g. int constants, dim vars, and integer operations on these)

            # coerce to ShapeExpr when fields are all PrimExprs
            if all([isinstance(f, tir.PrimExpr) for f in fields]):
                return relax.ShapeExpr(fields, span=self.to_tvm_span(expr.span))
            return relay.Tuple(fields, span=self.to_tvm_span(expr.span))

        elif isinstance(expr, ast.Var):
            var_name = expr.id.name
            if _is_registered(var_name, op_set=self._registered_ops):
                return relay.op.get(var_name)
            if var_name in self.scope:
                return self.scope[var_name]
            # NOTE: this is a "hack" to get around Python eagerly parsing class method decorators
            #       first (meaning we need to resolve them after the functions are parsed). These
            #       GlobalVars need to be resolved using string equality only.
            return relay.GlobalVar(var_name)

        elif isinstance(expr, ast.Constant):
            # FIXME(@altanh): use internal representation that doesn't have precision limits here
            if isinstance(expr.value, int):
                return tir.IntImm("int64", expr.value, self.to_tvm_span(expr.span))
            elif isinstance(expr.value, float):
                return tir.FloatImm("float32", expr.value, self.to_tvm_span(expr.span))
            elif isinstance(expr.value, str):
                # FIXME(@altanh): using StringImm seems to cause problems, but this loses span
                return expr.value
            elif expr.value is None:
                return None
            else:
                return relax.const(expr.value)

        elif isinstance(expr, ast.ArrayLiteral):
            return self.parse_array_literal(expr)

        elif isinstance(expr, ast.Op):
            # TODO(@altanh): might need to generalize from ArithmeticOp if we decide to support
            #                array slicing syntax
            try:
                return ArithmeticOp(expr.name)
            except ValueError:
                self.report_error(f"unsupported built-in operator: {expr.name}", expr.span)
        else:
            self.report_error(f"unsupported expression: {expr}", expr.span)

    def transform_Subscript(self, expr):
        """Array access visitor."""

        symbol = self.transform(expr.params[0])
        if symbol is None:
            self.report_error(
                f"Variable {expr.params[0].id.name} is not defined.", expr.params[0].span
            )
        indexes = [self.transform(x) for x in expr.params[1].values]
        if isinstance(symbol, relax.expr.Var):
            if len(indexes) > 1:
                self.report_error(
                    "Only a single index can be provided when indexing into a `var`.",
                    expr.params[1].span,
                )
            index = indexes[0].value
            if not isinstance(index, (tvm.tir.PrimExpr, int)):
                self.report_error(
                    "Var load index should be an int or PrimExpr, but it is a" + type(index),
                    expr.span,
                )
            return call_with_error_reporting(
                self.report_error,
                expr.span,
                relax.TupleGetItem,
                symbol,
                index,
            )
        elif isinstance(symbol, tvm.tir.expr.Var):
            if symbol.dtype == "handle":
                self.report_error(
                    "Cannot read directly from a handle, use `T.match_buffer` "
                    "to create a buffer to read from.",
                    expr.params[0].span,
                )
            if len(indexes) > 1:
                self.report_error(
                    "Only a single index can be provided when indexing into a `var`.",
                    expr.params[1].span,
                )
            index = indexes[0]
            if not isinstance(index, (tvm.tir.PrimExpr, int)):
                self.report_error(
                    "Var load index should be an int or PrimExpr, but it is a" + type(index),
                    expr.span,
                )

            return call_with_error_reporting(
                self.report_error,
                expr.span,
                tvm.tir.Load,
                "float32",
                symbol,
                index,
                True,
                span=tvm_span_from_synr(expr.span),
            )
        elif isinstance(symbol, tvm.tir.Buffer):
            return BufferSlice(
                symbol, indexes, self.report_error, span=tvm_span_from_synr(expr.span)
            )
        elif isinstance(symbol, tvm.container.Array):
            if len(indexes) > 1:
                self.report_error(
                    "Array access should be one-dimension access, but the indices are "
                    + str(indexes),
                    expr.span,
                )
            index = indexes[0]
            if not isinstance(index, (int, tvm.tir.expr.IntImm)):
                self.report_error(
                    "Array access index expected int or IntImm, but got " + type(index),
                    expr.span,
                )
            if int(index) >= len(symbol):
                self.report_error(
                    f"Array access out of bound, size: {len(symbol)}, got index {index}.",
                    expr.span,
                )
            return symbol[int(index)]
        else:
            self.report_error(
                f"Cannot subscript from a {type(symbol).__name__}.",
                expr.params[0].span,
            )

    def transform_block(self, block: ast.Block) -> relax.SeqExpr:
        """Transforms the given synr block to a Relax SeqExpr (sequence of Blocks with a final
        expression).

        Parameters
        ----------
        block : ast.Block
            The input synr block

        Returns
        -------
        relax.SeqExpr
            The parsed SeqExpr
        """
        # a block of statements needs to be converted to a SeqExpr of binding blocks
        blocks = []
        current_block = []
        for stmt in block.stmts[:-1]:
            parsed_stmt = self.transform_stmt(stmt)
            if isinstance(parsed_stmt, relax.DataflowBlock):
                if current_block:
                    # FIXME(@altanh): need to manually construct span start & end
                    blocks.append(relax.BindingBlock(current_block, self.to_tvm_span(stmt.span)))
                    current_block = []
                blocks.append(parsed_stmt)
            elif isinstance(parsed_stmt, (relax.Function, tir.PrimFunc)):
                func_var = self.decl_var(stmt.name, None, None, stmt.span)
                current_block.append(
                    relax.VarBinding(func_var, parsed_stmt, self.to_tvm_span(stmt.span))
                )
            else:
                assert isinstance(
                    parsed_stmt, relax.Binding
                ), "Expected relax.Binding, but got " + str(type(parsed_stmt))
                current_block.append(parsed_stmt)
        if len(current_block) > 0:
            blocks.append(relax.BindingBlock(current_block, self.to_tvm_span(block.stmts[-1].span)))

        ret_stmt = block.stmts[-1]
        if not isinstance(ret_stmt, ast.Return):
            self.report_error(
                "block must end with a returned expression",
                ret_stmt.span,
            )
        ret_expr = self.transform_stmt(ret_stmt)

        # only a call node in the function body
        if isinstance(ret_expr, relax.Call) and len(blocks) == 0:
            return ret_expr

        # return a defined inner function
        if (
            len(blocks) > 0
            and isinstance(blocks[-1].bindings[-1].value, relax.Function)
            and hasattr(ret_expr, "name_hint")
            and ret_expr.name_hint == blocks[-1].bindings[-1].var.name_hint
        ):
            return blocks[-1].bindings[-1].value

        return relax.SeqExpr(blocks, ret_expr, self.to_tvm_span(block.span))


class RelaxDiagnosticContext(synr.DiagnosticContext):
    """Relax diagnostic context"""

    def __init__(self, ir_mod):
        self.tvm_diag_ctx = diagnostics.DiagnosticContext(ir_mod, diagnostics.get_renderer())
        self.str_to_source_name = {}
        self._render_on_error = False

    def to_tvm_span(self, ast_span: ast.Span) -> tvm.ir.Span:
        return tvm.ir.Span(
            self.str_to_source_name[ast_span.filename],
            ast_span.start_line,
            ast_span.end_line,
            ast_span.start_column,
            ast_span.end_column,
        )

    def add_source(self, name: str, source: str) -> None:
        """Add a file with source code to the context. This will be called
        before any call to :py:func:`emit` that contains a span in this
        file.
        """
        src_name = self.tvm_diag_ctx.module.source_map.add(name, source)
        self.str_to_source_name[name] = src_name

    def emit(self, level: str, message: str, span: tvm.ir.Span) -> None:
        """Called when an error has occured."""
        if isinstance(span, ast.Span):
            span = self.to_tvm_span(span)

        if level == "error":
            level = diagnostics.DiagnosticLevel.ERROR
        elif level == "bug":
            level = diagnostics.DiagnosticLevel.BUG
        elif level == "warning":
            level = diagnostics.DiagnosticLevel.WARNING
        else:
            level = "error"

        assert span, "Span must not be null"
        assert isinstance(span, tvm.ir.Span), "Expected tvm.ir.Span, but got " + str(type(span))

        diag = diagnostics.Diagnostic(level, span, message)

        self.tvm_diag_ctx.emit(diag)
        if self._render_on_error:
            self.render()

    def render(self) -> Optional[Any]:
        """Render out all error messages. Can either return a value or raise
        and execption.
        """
        self.tvm_diag_ctx.render()


# def script(f) -> Union[relax.Function, Callable[[], tvm.IRModule]]:
#     """Parses the decorated Relax function or module (in Relax IR) to a Relax AST.

#     Parameters
#     ----------
#     f : Union[function, class]
#         The function or class to be parsed, written in the Relax IR.

#     Returns
#     -------
#     Union[relax.Function, IRModule]
#         The parsed Relax function or IRModule factory (which returns the parsed IRModule when
#         called).
#     """
#     diag_ctx = tvm.script.diagnostics.TVMDiagnosticCtx()
#     ast = synr.to_ast(f, diag_ctx)
#     mod = RelaxTransformer().do_transform(ast, diag_ctx)
#     if isinstance(mod, tvm.IRModule):
#         return lambda: mod
#     return mod


def from_source(
    input_func: Union[str, Callable],
    relax_prefix: Optional[List[str]] = None,
    tir_prefix: Optional[List[str]] = None,
) -> Union[relax.Function, IRModule]:
    """Parse function or string into a Relax Function or IRModule.

    If possible, pass the TVM script in as a function so that line numbers and
    filename will be accurate.

    Parameters
    ----------
    input_func : Union[str, Callable]
        The python function to be parsed.

    relax_prefix : Optional[List[str]]
        The relax prefix list. Only works for str input, default by "relax" and "R".

    tir_prefix : Optional[List[str]]
        The tir prefix list. Only works for str input, default by "tir" and "T".

    Returns
    -------
    output : Union[Function, IRModule]
        The relax Function or IRModule.
    """
    metadata = None
    if isinstance(input_func, str) and "b64ndarrays" in input_func:
        input_func, metadata = metadata_partitioner(input_func)

    mod = IRModule(attrs=metadata)
    if isinstance(input_func, str):
        relax_prefix = ["R", "relax"] if relax_prefix is None else relax_prefix
        tir_prefix = ["T", "tir"] if tir_prefix is None else tir_prefix
        return synr.to_ast(
            input_func, RelaxDiagnosticContext(mod), RelaxTransformer(mod, relax_prefix, tir_prefix)
        )
    elif inspect.isfunction(input_func):
        env: Dict[str, Any] = input_func.__globals__
        relax_prefix = [key for key in env.keys() if env[key] == relax_namespace]
        tir_prefix = [key for key in env.keys() if env[key] == tir_namespace]
        return synr.to_ast(
            input_func, RelaxDiagnosticContext(mod), RelaxTransformer(mod, relax_prefix, tir_prefix)
        )
    else:
        raise TypeError("Only function definitions are supported.")


# def fromtext(source: str, source_name: str = "from_string")
# -> Union[relax.Function, tvm.IRModule]:
#     """Parses the given input string (in the Relax text format) to a Relax AST.

#     Parameters
#     ----------
#     source : str
#         The input source string. It should be either a decorated Python class or function.
#     source_name : str, optional
#         A descriptive name for error reporting, by default "from_string".

#     Returns
#     -------
#     Union[relax.Function, IRModule]
#         The parsed Relax function or IRModule factory (which returns the parsed IRModule when
#         called).
#     """
#     # TODO(@altanh): actually use source_name somewhere?
#     diag_ctx = tvm.script.diagnostics.TVMDiagnosticCtx()
#     ast = synr.to_ast(source, diag_ctx)
#     mod = RelaxTransformer().do_transform(ast, diag_ctx)
#     if isinstance(mod, tvm.IRModule):
#         return lambda: mod
#     return mod


def pretty_print(node, show_meta_data=False):
    """Prints the given Relax IR node in the Relax text format.

    Parameters
    ----------
    node : Union[relax.Type, relax.Expr, relax.Binding, relax.BindingBlock]
        The Relax IR node to print.

    show_meta_data : bool
        Whether to include meta data section in the text
        if there is meta data.
    """
    print(tvm.script._ffi_api.AsRelaxScript(node, show_meta_data))


# TODO(@altanh): printer stuff should probably live elsewhere?
def astext(node, show_meta_data=False) -> str:
    """Returns the Relax text format representation of the given Relax IR node.

    Parameters
    ----------
    node : Union[relax.Type, relax.Expr, relax.Binding, relax.BindingBlock]
        The Relax IR node to print.

    show_meta_data : bool
        Whether to include meta data section in the text
        if there is meta data.

    Returns
    -------
    relax_text: str
        The text format representation of the given Relax IR node.
        If show_meta_data is True, the meta data section will be printed in the beginning
        of the the return string.
    """
    return tvm.script._ffi_api.AsRelaxScript(node, show_meta_data)
