from __future__ import annotations

import inspect
from typing import TypeVar, Generic, Union, Dict, List, Tuple, Optional
from io import StringIO
from enum import Enum

import tvm
import tvm.script
from tvm.ir.module import IRModule
from tvm.relay.base import Id
from tvm.ir import diagnostics
from tvm import tir

import numpy as np

import synr
from synr import ast, Transformer
from synr.diagnostic_context import DiagnosticContext
from tvm.relay.op.strategy.generic import conv1d_strategy

import tvm.relay as relay
import tvm.relax as rx


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


def _tir_from_synr(
    synr_ast: ast.Node, diag_ctx: tvm.script.diagnostics.TVMDiagnosticCtx
) -> tir.PrimFunc:
    """Parses the given synr AST using the TVMScript parser to a PrimFunc.

    Parameters
    ----------
    synr_ast : ast.Node
        The synr AST to be parsed.
    diag_ctx : tvm.script.diagnostics.TVMDiagnosticCtx
        The diagnostic context for TVMScript parser error reporting.

    Returns
    -------
    tir.PrimFunc
        The parsed TIR PrimFunc.
    """
    parser = tvm.script.parser.TVMScriptParser(synr_ast.span.start_line)
    return parser.do_transform(synr_ast, diag_ctx)


# NOTE: call_dps is an actual registered operator
class SpecialOp(Enum):
    """Relax operators that have special semantics handled by the parser."""

    MATCH_SHAPE = "relax.match_shape"
    CALL_PACKED = "relax.call_packed"
    DATAFLOW = "relax.dataflow"
    DATAFLOW_OUTPUT = "relax.output"


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
    def __init__(self):
        super().__init__()
        self.module = tvm.IRModule()
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
        return self._diagnostic_context.to_tvm_span(self._diagnostic_context.source_name, span)

    def report_error(self, msg: str, span: ast.Span):
        """Helper method for emitting and immediately rendering an error.

        Parameters
        ----------
        msg : str
            The error message
        span : ast.Span
            The span to report the error at
        """
        self._diagnostic_context.emit("error", msg, span)
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

    @property
    def scope(self):
        """Returns the current definition scope.

        Returns
        -------
        Dict[str, Union[rx.Var, tir.Var]]
            The scope of all currently defined variables (Relax and TIR).
        """
        return self._scopes[-1]

    def decl_var(
        self,
        name: str,
        type_annotation: Optional[rx.Type],
        shape: Optional[rx.Expr],
        span: ast.Span,
        is_dataflow: bool = False,
    ) -> rx.Var:
        """Introduces a variable with the given name and annotations to the current scope.

        Parameters
        ----------
        name : str
            The name of the variable
        type_annotation : Optional[rx.Type]
            The type annotation
        shape : Optional[rx.Expr]
            The shape annotation
        span : ast.Span
            The span where the variable is declared

        Returns
        -------
        Union[rx.Var, rx.DataflowVar]
            The declared variable
        """
        if name in self.scope:
            # TODO(@altanh): maybe emit an error at the declaration site and report it together
            self.report_error("variable has already been declared in the current scope", span)
        if is_dataflow:
            var = rx.DataflowVar(name, shape, type_annotation, self.to_tvm_span(span))
        else:
            var = rx.Var(name, shape, type_annotation, self.to_tvm_span(span))
        self.scope[name] = var
        return var

    def transform_type(self, ty: ast.Type, bind_free_vars: bool) -> Tuple[rx.Type, rx.Expr]:
        """Transforms the given synr type annotation to a Relax type and shape expression.

        Parameters
        ----------
        ty : ast.Type
            The synr type
        bind_free_vars : bool
            Whether or not the shape annotation can introduce new dimension variables

        Returns
        -------
        Tuple[rx.Type, rx.Expr]:
            The corresponding Relax type and shape expression
        """
        if ty is None:
            return (None, None)

        span = self.to_tvm_span(ty.span)

        # simple annotation with no type arguments
        if isinstance(ty, ast.TypeVar):
            if ty.id.name == "Tensor":
                return (rx.DynTensorType(rank=-1, dtype=None, span=span), None)
            elif ty.id.name == "Shape":
                return (rx.ShapeType(span), None)
            elif ty.id.name == "Dim":
                return (rx.DimType(span), None)
            else:
                self.report_error("unknown type in annotation", span)

        # annotation with type arguments/shape annotation
        if isinstance(ty, ast.TypeApply):
            if ty.id.name == "Tensor":
                # TODO(@altanh): forgetting dtype like "Tensor[(n, m)]" ends up getting parsed as
                #                Tensor[n, m] which makes correct errors difficult here...
                if len(ty.params) != 2:
                    self.report_error(
                        "Tensor type annotations must have 2 fields (shape and dtype)",
                        span,
                    )

                shape_annotation, dtype_annotation = ty.params
                shape, dtype, rank = None, None, -1

                # parse the shape annotation
                if isinstance(shape_annotation, ast.TypeVar):
                    if shape_annotation.id.name != "_":
                        # TODO(@altanh): handle variable annotations, e.g. x: Tensor[my_shape, _]
                        self.report_error(
                            "variable Tensor shape annotations not yet supported",
                            shape_annotation.span,
                        )
                    else:
                        # FIXME(@altanh): use a special node for unknown shape vs no shape?
                        pass  # shape = None
                elif isinstance(shape_annotation, ast.TypeTuple):
                    # the syntax for fixed rank k but unknown/unmatched shape is a tuple of length
                    # k, where each element is "_" (e.g. "(_, _)" for rank 2)
                    is_unmatched = all(
                        map(
                            lambda v: isinstance(v, ast.Var) and v.id.name == "_",
                            shape_annotation.values,
                        )
                    )
                    if len(shape_annotation.values) > 0 and is_unmatched:
                        rank = len(shape_annotation.values)
                    else:
                        shape = rx.ShapeExpr(
                            self.parse_shape(shape_annotation, bind_free_vars),
                            span=self.to_tvm_span(shape_annotation.span),
                        )
                        rank = len(shape)
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
                else:
                    self.report_error(
                        "Tensor dtype annotations must be concrete or erased",
                        dtype_annotation.span,
                    )

                return (rx.DynTensorType(rank=rank, dtype=dtype, span=span), shape)
            elif ty.id.name == "Tuple":
                field_types = []
                field_shapes = []
                for field in ty.params:
                    fty, fsh = self.transform_type(field, bind_free_vars=False)
                    field_types.append(fty)
                    field_shapes.append(fsh)
                return (relay.TupleType(field_types, span), None)
            # TODO(@altanh): other types with args, e.g. Ref[T], func types
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
                # introduce TIR variable to scope, e.g. for func params or rx.call_packed
                var = tir.Var(var_name, "int32", self.to_tvm_span(expr.span))
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
            return tir.const(expr.value, "int32", self.to_tvm_span(expr.span))

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
                self.module[global_var] = self.transform_function(func, is_global=True)
            return self.module
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
        strs.append(attr.id.name)
        result = ".".join(reversed(strs))
        return result

    def transform_function(self, func: ast.Function, is_global: bool = False) -> rx.Function:
        """Transforms the given synr Function to a Relax Function.

        Parameters
        ----------
        func : ast.Function
            The input synr Function
        is_global : bool, optional
            Whether or not the input function is global/module-level, by default False

        Returns
        -------
        rx.Function
            The parsed Relax Function
        """
        if (
            len(func.decorators) == 1
            and self._parse_attrs_to_str(func.decorators[0]) == "tvm.script.tir"
        ):
            return _tir_from_synr(func, self._diagnostic_context)

        with self.new_scope():
            params = []
            for param in func.params:
                ty, shape = self.transform_type(param.ty, bind_free_vars=True)
                param = self.decl_var(param.name, ty, shape, param.span)
                params.append(param)
            new_body = self.transform_block(func.body)
            ret_type, _ = self.transform_type(func.ret_type, bind_free_vars=False)

        func_name = rx.GlobalVar(func.name) if is_global else None
        return rx.Function(
            params, new_body, ret_type, name=func_name, span=self.to_tvm_span(func.span)
        )

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
            call_op = self.transform_expr(stmt.rhs.func_name)
        return call_op == SpecialOp.MATCH_SHAPE

    def parse_binding(self, stmt: ast.Stmt, is_dataflow: bool = False) -> rx.Binding:
        """Parses the input synr statement to the corresponding Relax binding.

        Parameters
        ----------
        stmt : ast.Stmt
            The input synr statement (either an assignment or a unassigned call)
        is_dataflow : bool, optional
            Whether or not the binding is in a dataflow block, by default False

        Returns
        -------
        rx.Binding
            The parsed Relax binding
        """
        assert isinstance(stmt, (ast.Assign, ast.UnassignedCall))
        if self.is_match_shape(stmt):
            return self.parse_shape_binding(stmt, is_dataflow=is_dataflow)
        else:
            assert isinstance(stmt, ast.Assign)
            return self.parse_var_binding(stmt, is_dataflow=is_dataflow)

    def parse_shape_binding(self, stmt: ast.Stmt, is_dataflow: bool = False) -> rx.MatchShape:
        """Parses the input synr statement to a Relax shape binding.

        Parameters
        ----------
        stmt : ast.Stmt
            The input synr statement
        is_dataflow : bool, optional
            Whether or not the bound variable (if any) is a dataflow variable, by default False

        Returns
        -------
        rx.MatchShape
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
            if not isinstance(stmt.lhs, ast.Var):
                self.report_error(
                    "the left hand side of a binding must be a variable", stmt.lhs.span
                )
            var = stmt.lhs
            call = stmt.rhs

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

        return rx.MatchShape(value, pattern, var, self.to_tvm_span(stmt.span))

    def parse_var_binding(self, stmt: ast.Assign, is_dataflow=False) -> rx.VarBinding:
        """Parses the input synr assignment to a Relax variable binding.

        Parameters
        ----------
        stmt : ast.Assign
            The input synr assignment
        is_dataflow : bool, optional
            Whether or not the bound variable is a dataflow variable, by default False

        Returns
        -------
        rx.VarBinding
            The parsed Relax variable binding
        """
        if not isinstance(stmt.lhs, ast.Var):
            self.report_error(
                "the left hand side of a binding must be a variable",
                stmt.lhs.span,
            )
        rhs = self.transform_expr(stmt.rhs)
        # an ExternFunc call comes from call_packed
        if isinstance(rhs, relay.Call) and isinstance(rhs.op, rx.ExternFunc):
            bind_free_vars = True
        else:
            bind_free_vars = False
        ty, shape = self.transform_type(stmt.ty, bind_free_vars)
        lhs = self.decl_var(stmt.lhs.id.name, ty, shape, stmt.lhs.span, is_dataflow=is_dataflow)
        return rx.VarBinding(lhs, rhs, self.to_tvm_span(stmt.span))

    # Stmts:
    # - Assert: probably unsupported for now
    # - Assign: VarBinding
    # - For: ??
    # - If: IfThenElse, must check no empty false branch
    # - Return: just the returned expression, must terminate blocks? (special case if-else)
    # - UnassignedCall: match_shape
    # - With: rx.dataflow
    def transform_stmt(self, stmt: ast.Stmt) -> Union[rx.Expr, rx.Binding, rx.DataflowBlock]:
        """Transforms the given synr statement to the corresponding Relax node.

        Parameters
        ----------
        stmt : ast.Stmt
            The input synr statement

        Returns
        -------
        Union[rx.Expr, rx.Binding, rx.DataflowBlock]
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
            if not isinstance(true_assign, ast.Assign) or not isinstance(true_assign.lhs, ast.Var):
                self.report_error(
                    "each branch of an if-else statement must end in a variable assignment",
                    true_assign.span,
                )
            if not isinstance(false_assign, ast.Assign) or not isinstance(
                false_assign.lhs, ast.Var
            ):
                self.report_error(
                    "each branch of an if-else statement must end in a variable assignment",
                    false_assign.span,
                )
            union_span = ast.Span.union([true_assign.span, false_assign.span])
            if true_assign.lhs.id.name != false_assign.lhs.id.name:
                self.report_error(
                    "the final assignment of both branches must have the same variable",
                    union_span,
                )

            var_name = true_assign.lhs.id.name

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
            return rx.VarBinding(var, ite_expr, self.to_tvm_span(union_span))

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
            func_var = self.decl_var(stmt.name, None, None, stmt.span)
            return rx.VarBinding(func_var, func, self.to_tvm_span(stmt.span))

        else:
            self.report_error(
                "unsupported statement",
                stmt.span,
            )

    def parse_dataflow(self, block: ast.Block) -> rx.DataflowBlock:
        """Parses the input synr block to a Relax dataflow block.

        Parameters
        ----------
        block : ast.Block
            The input synr block

        Returns
        -------
        rx.DataflowBlock
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
                    and binding_stmt.lhs.id.name not in output_var_names
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

        return rx.DataflowBlock(bindings, self.to_tvm_span(block.span))

    def parse_attr(self, expr: ast.Attr) -> rx.Expr:
        """Parses the given synr Attr node to a Relax expression.

        Parameters
        ----------
        expr : ast.Attr
            The synr Attr node to be parsed.

        Returns
        -------
        rx.Expr
            The parsed expression.
        """
        if expr.field.name == "shape":
            obj = self.transform_expr(expr.object)
            attrs = tvm.ir.attrs.make_node("relay.attrs.ShapeOfAttrs", dtype="int32")
            return relay.Call(
                relay.op.get("shape_of"), [obj], attrs=attrs, span=self.to_tvm_span(expr.span)
            )
        else:
            # assume it's a hierarchical op identifier (e.g. nn.softmax, relax.call_dps)
            op_name = self._parse_attrs_to_str(expr)
            # NOTE: at least for now, all special operators are namespaced
            try:
                return SpecialOp(op_name)
            except ValueError:
                # TODO(@altanh): maybe diagnostics here in case this fails?
                return relay.op.get(op_name)

    def parse_call(self, expr: ast.Call) -> Union[tir.PrimExpr, rx.Expr]:
        """Parses the given synr Call node to a Relax expression or PrimExpr.

        Parameters
        ----------
        expr : ast.Call
            The synr Call node to be parsed.

        Returns
        -------
        Union[tir.PrimExpr, rx.Expr]
            The parsed expression. It will be a PrimExpr if expr is an arithmetic operation on
            PrimExprs.
        """
        op = self.transform_expr(expr.func_name)

        if op == SpecialOp.CALL_PACKED:
            if len(expr.params) != 2:
                self.report_error(
                    op.value + " takes an extern function name and a tuple of arguments",
                    expr.span,
                )
            extern_func = expr.params[0]
            if not (isinstance(extern_func, ast.Constant) and isinstance(extern_func.value, str)):
                self.report_error(
                    "the first argument of " + op.value + " must be the extern function name",
                    extern_func.span,
                )
            op = rx.ExternFunc(extern_func.value, self.to_tvm_span(extern_func.span))
            args = [self.transform_expr(expr.params[1])]

        elif isinstance(op, ArithmeticOp):
            args = [self.transform_expr(arg) for arg in expr.params]
            if all([isinstance(arg, tir.PrimExpr) for arg in args]):
                return PRIMEXPR_ARITHMETIC_OP_MAP[op](*args, span=self.to_tvm_span(expr.span))
            # otherwise it's just a normal Relax operator call
            op = RELAX_ARITHMETIC_OP_MAP[op]

        elif isinstance(op, tvm.ir.Op):
            args = [self.transform_expr(arg) for arg in expr.params]
            # check call arity eagerly
            if op.num_inputs != -1 and len(args) != op.num_inputs:
                self.report_error(
                    f"{op.name} expects {op.num_input} arguments but got {len(args)}", expr.span
                )
            if op.name == "relax.call_dps" and isinstance(args[1], str):
                # extern function call case: rewrite identifier to an ExternFunc
                args[1] = rx.ExternFunc(args[1], self.to_tvm_span(expr.params[1].span))

        elif isinstance(op, relay.Expr):
            args = [self.transform_expr(arg) for arg in expr.params]

        else:
            self.report_error(f"unsupported function in call: {op}", expr.func_name.span)

        # parse call attributes if applicable
        if isinstance(op, rx.ExternFunc) or (isinstance(op, tvm.ir.Op) and op.attrs_type_key != ""):
            attrs_type_key = "DictAttrs" if isinstance(op, rx.ExternFunc) else op.attrs_type_key
            kwargs = {}
            for key, val in expr.keyword_params.items():
                assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                # TODO(@altanh): might need separate attribute parsing eventually
                kwargs[key.value] = self.transform_expr(val)
            attrs = tvm.ir.attrs.make_node(attrs_type_key, **kwargs)
        else:
            attrs = None

        return relay.Call(op, args, attrs=attrs, span=self.to_tvm_span(expr.span))

    # Exprs:
    # - ArrayLiteral: unsupported for now?
    # - Attr: use for .shape, and intrinsic/special operator namespace
    # - Call
    # - Constant
    # - DictLiteral: unsupported for now
    # - Slice: unsupported for now, could desugar to slice op
    # - Tuple
    # - Var
    def transform_expr(self, expr: ast.Expr) -> rx.Expr:
        """Transforms the input synr expression to a Relax expression.

        Parameters
        ----------
        expr : ast.Expr
            The input synr

        Returns
        -------
        rx.Expr
            The corresponding Relax expression
        """
        if isinstance(expr, ast.Attr):
            return self.parse_attr(expr)

        elif isinstance(expr, ast.Call):
            return self.parse_call(expr)

        elif isinstance(expr, ast.Tuple):
            fields = [self.transform_expr(field) for field in expr.values]

            # TODO(@altanh): this check might be too weak; we really only accept integral PrimExprs
            #                (e.g. int constants, dim vars, and integer operations on these)

            # coerce to ShapeExpr when fields are all PrimExprs
            if all([isinstance(f, tir.PrimExpr) for f in fields]):
                return rx.ShapeExpr(fields, span=self.to_tvm_span(expr.span))
            return relay.Tuple(fields, span=self.to_tvm_span(expr.span))

        elif isinstance(expr, ast.Var):
            var_name = expr.id.name
            if _is_registered(var_name, op_set=self._registered_ops):
                return relay.op.get(var_name)
            if var_name not in self.scope:
                self.report_error("undefined variable", expr.span)
            return self.scope[var_name]

        elif isinstance(expr, ast.Constant):
            # FIXME(@altanh): use internal representation that doesn't have precision limits here
            if isinstance(expr.value, int):
                return tir.IntImm("int32", expr.value, self.to_tvm_span(expr.span))
            elif isinstance(expr.value, float):
                return tir.FloatImm("float32", expr.value, self.to_tvm_span(expr.span))
            elif isinstance(expr.value, str):
                # FIXME(@altanh): using StringImm seems to cause problems, but this loses span
                return expr.value
            elif expr.value is None:
                return None
            else:
                self.report_error(
                    f"unsupported constant expression: {expr}",
                    expr.span,
                )

        elif isinstance(expr, ast.Op):
            # TODO(@altanh): might need to generalize from ArithmeticOp if we decide to support
            #                array slicing syntax
            try:
                return ArithmeticOp(expr.name)
            except ValueError:
                self.report_error(f"unsupported built-in operator: {expr.name}", expr.span)

        else:
            self.report_error(f"unsupported expression: {expr}", expr.span)

    def transform_block(self, block: ast.Block) -> rx.SeqExpr:
        """Transforms the given synr block to a Relax SeqExpr (sequence of Blocks with a final
        expression).

        Parameters
        ----------
        block : ast.Block
            The input synr block

        Returns
        -------
        rx.SeqExpr
            The parsed SeqExpr
        """
        # a block of statements needs to be converted to a SeqExpr of binding blocks
        blocks = []
        current_block = []
        for stmt in block.stmts[:-1]:
            parsed_stmt = self.transform_stmt(stmt)
            if isinstance(parsed_stmt, rx.DataflowBlock):
                if current_block:
                    # FIXME(@altanh): need to manually construct span start & end
                    blocks.append(rx.BindingBlock(current_block, self.to_tvm_span(stmt.span)))
                    current_block = []
                blocks.append(parsed_stmt)
            else:
                assert isinstance(parsed_stmt, rx.Binding)
                current_block.append(parsed_stmt)
        if len(current_block) > 0:
            blocks.append(rx.BindingBlock(current_block, self.to_tvm_span(stmt.span)))

        ret_stmt = block.stmts[-1]
        if not isinstance(ret_stmt, ast.Return):
            self.report_error(
                "block must end with a returned expression",
                ret_stmt.span,
            )
        ret_expr = self.transform_stmt(ret_stmt)

        return rx.SeqExpr(blocks, ret_expr, self.to_tvm_span(block.span))


# class TVMDiagnosticContext(synr.DiagnosticContext):
#     def __init__(self, tvm_diag_ctx):
#         self.tvm_diag_ctx = tvm_diag_ctx
#         self.str_to_source_name = {}

#     def add_source(self, name: str, source: str) -> None:
#         """Add a file with source code to the context. This will be called
#         before any call to :py:func:`emit` that contains a span in this
#         file.
#         """
#         src_name = self.tvm_diag_ctx.module.source_map.add(name, source)
#         self.str_to_source_name[name] = src_name

#     def emit(self, level: str, message: str, span: tvm.ir.Span) -> None:
#         """Called when an error has occured."""

#         if level == "error":
#             level = diagnostics.DiagnosticLevel.ERROR
#         elif level == "bug":
#             level = diagnostics.DiagnosticLevel.BUG
#         elif level == "warning":
#             level = diagnostics.DiagnosticLevel.WARNING
#         else:
#             level = "error"

#         assert span, "Span must not be null"
#         assert isinstance(span, tvm.ir.Span), "Expected tvm.ir.Span, but got " + str(type(span))

#         diag = diagnostics.Diagnostic(level, span, message)

#         self.tvm_diag_ctx.emit(diag)

#     def render(self) -> Optional[Any]:
#         """Render out all error messages. Can either return a value or raise
#         and execption.
#         """
#         self.tvm_diag_ctx.render()


def script(f) -> Union[rx.Function, tvm.IRModule]:
    """Parses the decorated Relax function or module (in Relax IR) to a Relax AST.

    Parameters
    ----------
    f : Union[function, class]
        The function or class to be parsed, written in the Relax IR.

    Returns
    -------
    Union[rx.Function, IRModule]
        The parsed Relax function or IRModule.
    """
    diag_ctx = tvm.script.diagnostics.TVMDiagnosticCtx()
    ast = synr.to_ast(f, diag_ctx)
    return RelaxTransformer().do_transform(ast, diag_ctx)


def fromtext(source: str, source_name: str = "from_string") -> Union[rx.Function, tvm.IRModule]:
    """Parses the given input string (in the Relax text format) to a Relax function or IRModule.

    Parameters
    ----------
    source : str
        The input source string. It should be either a decorated Python class or function.
    source_name : str, optional
        A descriptive name for error reporting, by default "from_string".

    Returns
    -------
    Union[rx.Function, IRModule]
        The parsed Relax function or IRModule.
    """
    # TODO(@altanh): actually use source_name somewhere?
    diag_ctx = tvm.script.diagnostics.TVMDiagnosticCtx()
    ast = synr.to_ast(source, diag_ctx)
    return RelaxTransformer().do_transform(ast, diag_ctx)


def pretty_print(node):
    """Prints the given Relax IR node in the Relax text format.

    Parameters
    ----------
    node : Union[rx.Type, rx.Expr, rx.Binding, rx.BindingBlock]
        The Relax IR node to print.
    """
    print(tvm.script._ffi_api.AsRelaxScript(node))


def astext(node) -> str:
    """Returns the Relax text format representation of the given Relax IR node.

    Parameters
    ----------
    node : Union[rx.Type, rx.Expr, rx.Binding, rx.BindingBlock]
        The Relax IR node to print.

    Returns
    -------
    str
        The text format representation of the given Relax IR node.
    """
    return tvm.script._ffi_api.AsRelaxScript(node)
