from __future__ import annotations

import inspect
from typing import TypeVar, Generic, Union, Dict, List, Tuple, Optional
from io import StringIO

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


# TODO: Replace with a real pretty print method once we have the real AST
def pretty_print(f):
    print(f)


def is_registered(op_name, op_set=None):
    if op_set is None:
        op_set = tvm.ir._ffi_api.ListOpNames()
    return op_name in op_set


def _tir_from_synr(synr_ast: ast.Node, diag_ctx: tvm.script.diagnostics.TVMDiagnosticCtx):
    parser = tvm.script.parser.TVMScriptParser(synr_ast.span.start_line)
    return parser.do_transform(synr_ast, diag_ctx)


class RelaxTransformer(Transformer):
    def __init__(self, definition_scope):
        super().__init__()
        self.definition_scope = definition_scope
        self.module = {}
        self._scopes = [{}]  # str -> Var
        self._registered_ops = set(tvm.ir._ffi_api.ListOpNames())  # cached

    def to_tvm_span(self, span: ast.Span) -> tvm.ir.Span:
        return self._diagnostic_context.to_tvm_span(self._diagnostic_context.source_name, span)

    def report_error(self, msg: str, span: ast.Span):
        self._diagnostic_context.emit("error", msg, span)
        self._diagnostic_context.render()

    def new_scope(self):
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
        type_annotation : Optional[rxType]
            The type annotation
        shape : Optional[rxExpr]
            The shape annotation
        span : ast.Span
            The span where the variable is declared

        Returns
        -------
        rxVar
            The declared variable
        """
        if name in self.scope:
            self.report_error("variable has already been declared in the current scope", span)
        if is_dataflow:
            var = rx.DataflowVar(name, shape, type_annotation, self.to_tvm_span(span))
        else:
            var = rx.Var(name, shape, type_annotation, self.to_tvm_span(span))
        self.scope[name] = var
        return var

    def transform_type(self, ty: ast.Type, allow_intro: bool) -> Tuple[rx.Type, rx.Expr]:
        """Transforms the given synr type annotation to a Relax type and shape expression.

        Parameters
        ----------
        ty : ast.Type
            The synr type
        allow_intro : bool
            Whether or not the shape annotation can introduce new dimension variables

        Returns
        -------
        Tuple[rxType, rxExpr]:
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
                if len(ty.params) != 2:
                    self.report_error(
                        "Tensor type annotations must have 2 fields (shape and dtype)",
                        span,
                    )

                shape_annotation, dtype_annotation = ty.params
                shape, dtype = None, None

                # parse the shape annotation
                if isinstance(shape_annotation, ast.TypeVar):
                    if shape_annotation.id.name != "_":
                        # TODO: handle variable annotations, e.g. x: Tensor[my_shape, _]
                        self.report_error(
                            "variable Tensor shape annotations not yet supported",
                            shape_annotation.span,
                        )
                    else:
                        # FIXME: use a special node for unknown shape vs no shape?
                        pass  # shape = None
                elif isinstance(shape_annotation, ast.TypeTuple):
                    shape = rx.ShapeExpr(
                        self.parse_shape(shape_annotation, allow_intro),
                        span=self.to_tvm_span(shape_annotation.span),
                    )
                else:
                    self.report_error(
                        "unsupported shape annotation",
                        shape_annotation.span,
                    )

                # parse the dtype annotation
                if isinstance(dtype_annotation, ast.TypeVar) and dtype_annotation.id.name == "_":
                    pass  # dtype = None
                elif isinstance(dtype_annotation, ast.TypeConstant):
                    dtype = dtype_annotation.value  # TODO: parse to TVM DType?
                else:
                    self.report_error(
                        "Tensor dtype annotations must be concrete or erased",
                        dtype_annotation.span,
                    )

                rank = len(shape) if shape is not None else -1
                return (rx.DynTensorType(rank=rank, dtype=dtype, span=span), shape)
            elif ty.id.name == "Tuple":
                field_types = []
                field_shapes = []
                for field in ty.params:
                    fty, fsh = self.transform_type(field, allow_intro=False)
                    field_types.append(fty)
                    field_shapes.append(fsh)
                return (relay.TupleType(field_types, span), None)
            # TODO: other types with args, e.g. Ref[T], func types
        self.report_error("invalid type", span)

    def parse_shape(
        self,
        shape_annotation: Union[ast.TypeTuple, ast.Tuple],
        allow_intro: bool,
    ) -> List[tir.PrimExpr]:
        """Parses the given shape annotation to a list of PrimExprs

        Parameters
        ----------
        shape_annotation : ast.TypeTuple
            The shape annotation in synr
        allow_intro : bool
            Whether or not the annotation can bind previously free variables

        Returns
        -------
        List[tir.PrimExpr]
            The parsed shape as a list of PrimExprs
        """
        return [self.parse_primexpr(field, allow_intro) for field in shape_annotation.values]

    def parse_primexpr(self, expr: ast.Expr, allow_intro: bool) -> tir.PrimExpr:
        """Parses the given expression to a PrimExpr

        Parameters
        ----------
        expr : ast.Expr
            The input expression
        allow_intro : bool
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
                    self.report_error(
                        "non-dimension variables cannot appear in dimension expressions",
                        expr.span,
                    )
                return var
            elif allow_intro:
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
        else:
            # TODO: parse (simple) PrimExprs
            self.report_error(
                "only dimension variable expressions are currently supported",
                expr.span,
            )

    def transform_module(self, mod: ast.Module) -> IRModule:
        for func_name in mod.funcs:
            func = mod.funcs[func_name]
            self.module[func_name] = self.transform_function(func, is_global=True)
        return self.module

    def transform_function(self, func: ast.Function, is_global=False) -> rx.Function:
        if (
            len(func.decorators) == 1
            and isinstance(func.decorators[0], ast.Var)
            and func.decorators[0].id.name == "tir"
        ):
            return _tir_from_synr(func, self._diagnostic_context)

        with self.new_scope():
            params = []
            for param in func.params:
                ty, shape = self.transform_type(param.ty, allow_intro=True)
                param = self.decl_var(param.name, ty, shape, param.span)
                params.append(param)
            new_body = self.transform_block(func.body)
            ret_type, _ = self.transform_type(func.ret_type, allow_intro=False)

        func_name = rx.GlobalVar(func.name) if is_global else None
        return rx.Function(
            params, new_body, ret_type, name=func_name, span=self.to_tvm_span(func.span)
        )

    def parse_binding(self, stmt: ast.Stmt, is_dataflow=False):
        assert isinstance(stmt, (ast.Assign, ast.UnassignedCall))
        if isinstance(stmt, ast.Assign):
            return self.parse_var_binding(stmt, is_dataflow=is_dataflow)
        else:
            return self.parse_shape_binding(stmt)

    def parse_shape_binding(self, stmt: ast.UnassignedCall):
        call: synr.ast.Call = stmt.call
        op = self.transform_expr(call.func_name)
        if op != relay.op.get("relax.match_shape"):
            self.report_error("the results of calls must be bound or used", stmt.span)
        if len(stmt.call.params) != 2:
            self.report_error("relax.match_shape takes exactly two arguments", stmt.span)

        lhs = stmt.call.params[0]
        rhs = stmt.call.params[1]

        rhs_expr = self.transform_expr(rhs)
        if not isinstance(lhs, ast.Tuple):
            self.report_error(
                "the pattern (lhs) of relax.match_shape must be a tuple",
                lhs.span,
            )
        lhs_expr = self.parse_shape(lhs, allow_intro=True)
        return rx.MatchShape(lhs_expr, rhs_expr, self.to_tvm_span(stmt.span))

    def parse_var_binding(self, stmt: ast.Assign, is_dataflow=False):
        if not isinstance(stmt.lhs, ast.Var):
            self.report_error(
                "the left hand side of a binding must be a variable",
                stmt.lhs.span,
            )
        # TODO: figure out proper way of doing this
        rhs = self.transform_expr(stmt.rhs)
        if isinstance(rhs, relay.Call) and rhs.op == relay.op.get("relax.call_packed"):
            allow_intro = True
        else:
            allow_intro = False
        ty, shape = self.transform_type(stmt.ty, allow_intro)
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
            # TODO: the spans here are all sorts of messed up, not sure how to fix
            ite_expr = relay.If(cond, true_branch, false_branch, self.to_tvm_span(stmt.span))
            # TODO: type and shape of return var
            var = self.decl_var(var_name, None, None, union_span)
            return rx.VarBinding(var, ite_expr, self.to_tvm_span(union_span))

        elif isinstance(stmt, ast.Return):
            return self.transform_expr(stmt.value)

        elif isinstance(stmt, ast.UnassignedCall):
            # FIXME: when we add ref support, ref_write can be unassigned
            return self.parse_shape_binding(stmt)

        elif isinstance(stmt, ast.With):
            if not isinstance(stmt.rhs, ast.Call):
                self.report_error("unsupported with block", stmt.span)

            call = stmt.rhs
            op = self.transform_expr(call.func_name)

            # TODO: perhaps this ought to be more general

            if op != relay.op.get("relax.dataflow"):
                self.report_error("unsupported with block type", call.span)
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
        assert len(block.stmts) > 0, "should never have an empty dataflow block"
        bindings = []
        output_vars = []

        with self.new_scope():
            # parse the return statement first to figure out which bindings assign normal Vars
            output_stmt = block.stmts[-1]
            if not isinstance(output_stmt, ast.Return):
                self.report_error(
                    "dataflow blocks must end with returning the output variables",
                    output_stmt.span,
                )

            ret_val = output_stmt.value
            if isinstance(ret_val, ast.Var):
                ret_val = ast.Tuple(values=[ret_val], span=ret_val.span)

            if not isinstance(ret_val, ast.Tuple) or any(
                [not isinstance(f, ast.Var) for f in ret_val.values]
            ):
                self.report_error(
                    "the returned values must be variables",
                    ret_val.span,
                )

            # output variables are bound to normal (not data flow) Vars
            output_var_names = {var.id.name for var in ret_val.values}

            for binding_stmt in block.stmts[:-1]:
                if not isinstance(binding_stmt, (ast.Assign, ast.UnassignedCall)):
                    self.report_error(
                        "only bindings are supported in dataflow blocks",
                        binding_stmt.span,
                    )
                is_match_shape = isinstance(binding_stmt, ast.UnassignedCall)
                is_dataflow = (
                    False if is_match_shape else (binding_stmt.lhs.id.name not in output_var_names)
                )
                binding = self.parse_binding(binding_stmt, is_dataflow=is_dataflow)
                bindings.append(binding)
                if not is_dataflow:
                    if is_match_shape:
                        for var in binding.pattern:
                            output_vars.append(var)
                    else:
                        output_vars.append(binding.var)

        # make output variables visible in parent scope
        for v in output_vars:
            # v could already be in scope if it was a previously bound dimension variable
            v_name = v.name if isinstance(v, tir.Var) else v.name_hint
            if v not in self.scope:
                self.scope[v_name] = v

        return rx.DataflowBlock(bindings, self.to_tvm_span(block.span))

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
        if isinstance(expr, ast.Attr):
            if expr.field.name == "shape":
                obj = self.transform_expr(expr.object)
                return relay.op.shape_of(obj)
            else:
                # assume it's a hierarchical op identifier (e.g. nn.softmax, rx.call_dps)
                op_name = []
                attr = expr
                while isinstance(attr, ast.Attr):
                    op_name.append(expr.field.name)
                    attr = attr.object
                if not isinstance(attr, ast.Var):
                    self.report_error("unsupported field access", expr.span)
                op_name.append(attr.id.name)
                op_name = ".".join(reversed(op_name))
                return relay.op.get(op_name)  # TODO: maybe diagnostics here in case this fails?

        if isinstance(expr, ast.Call):
            op = self.transform_expr(expr.func_name)
            args = [self.transform_expr(arg) for arg in expr.params]
            return relay.Call(op, args, span=self.to_tvm_span(expr.span))

        elif isinstance(expr, ast.Tuple):
            fields = [self.transform_expr(field) for field in expr.values]
            return relay.Tuple(fields, span=self.to_tvm_span(expr.span))

        elif isinstance(expr, ast.Var):
            var_name = expr.id.name
            if is_registered(var_name, op_set=self._registered_ops):
                return relay.op.get(var_name)
            if var_name not in self.scope:
                self.report_error("undefined variable", expr.span)
            return self.scope[var_name]

        else:
            self.report_error("unsupported expression", expr.span)

    def transform_block(self, block: ast.Block) -> rx.SeqExpr:
        # a block of statements needs to be converted to a SeqExpr of binding blocks
        blocks = []
        current_block = []
        for stmt in block.stmts[:-1]:
            parsed_stmt = self.transform_stmt(stmt)
            if isinstance(parsed_stmt, rx.DataflowBlock):
                if current_block:
                    # FIXME: span
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


class RelaxDecoratedFn:
    def __init__(self, fn_name, relax_module, diag_ctx):
        self.fn_name = fn_name
        self.module = relax_module
        self.diag_ctx = diag_ctx

    def __call__(self, *args):
        pretty_print(self.module[self.fn_name])
        # compiler = Compiler(self.diag_ctx, self.module, self.fn_name)
        # compiled_f = compiler.compile(execute=True)
        # # Actually compute needed buffer sizes.
        # out = tvm.nd.array(np.random.rand(10).astype('float32'))
        # compiled_f(*(list(args) + [out]))
        # return out


def script(f):
    # ir_module = tvm.IRModule({})
    # diag_ctx = diagnostics.DiagnosticContext(ir_module, diagnostics.get_renderer())
    diag_ctx = tvm.script.diagnostics.TVMDiagnosticCtx()
    ast = synr.to_ast(f, diag_ctx)
    definition_scope = inspect.getmodule(f)
    module = RelaxTransformer(definition_scope).do_transform(ast, diag_ctx)
    return RelaxDecoratedFn(f.__name__, module, diag_ctx)
