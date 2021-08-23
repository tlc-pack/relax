from __future__ import annotations

import inspect
from typing import TypeVar, Generic, Union, Dict, List, Tuple, Optional
from io import StringIO

import tvm
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


class RelaxTransformer(Transformer):
    def __init__(self, definition_scope, diag_ctx):
        super().__init__()
        self.definition_scope = definition_scope
        self.diag_ctx = diag_ctx
        self.module = {}
        self._scopes = [{}]  # str -> Var

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

    def tvm_span(self, span: synr.Span) -> tvm.ir.Span:
        """Converts the synr span to a TVM span

        Parameters
        ----------
        span : synr.Span
            The synr span

        Returns
        -------
        tvm.ir.Span
            The corresponding TVM span
        """
        src_name = self.diag_ctx.str_to_source_name[span.filename]
        tvm_span = tvm.ir.Span(
            src_name, span.start_line, span.end_line, span.start_column, span.end_column
        )
        return tvm_span

    def decl_var(
        self,
        name: str,
        type_annotation: Optional[rx.Type],
        shape: Optional[rx.Expr],
        span: tvm.ir.Span,
    ) -> Var:
        """Introduces a variable with the given name and annotations to the current scope.

        Parameters
        ----------
        name : str
            The name of the variable
        type_annotation : Optional[rxType]
            The type annotation
        shape : Optional[rxExpr]
            The shape annotation
        span : tvm.ir.Span
            The span where the variable is declared

        Returns
        -------
        rxVar
            The declared variable
        """
        if name in self.scope:
            self._diagnostic_context.emit(
                "error", "variable has already been declared in the current scope", span
            )
            self._diagnostic_context.render()
        var = rx.Var(name, type_annotation, shape, span)
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

        span = self.tvm_span(ty.span)

        # simple annotation with no type arguments
        if isinstance(ty, ast.TypeVar):
            if ty.id.name == "Tensor":
                return (rx.DynTensorType(rank=-1, dtype=None, span=span), None)
            elif ty.id.name == "Shape":
                return (rx.ShapeType(span), None)
            elif ty.id.name == "Dim":
                return (rx.DimType(span), None)
            else:
                self._diagnostic_context.emit("error", "unknown type in annotation", span)
                self._diagnostic_context.render()

        # annotation with type arguments/shape annotation
        if isinstance(ty, ast.TypeApply):
            if ty.id.name == "Tensor":
                if len(ty.params) != 2:
                    self._diagnostic_context.emit(
                        "error",
                        "Tensor type annotations must have 2 fields (shape and dtype)",
                        span,
                    )
                    self._diagnostic_context.render()

                shape_annotation, dtype_annotation = ty.params
                shape, dtype = None, None

                # parse the shape annotation
                if isinstance(shape_annotation, ast.TypeVar):
                    if shape_annotation.id.name != "_":
                        # TODO: handle variable annotations, e.g. x: Tensor[my_shape, _]
                        self._diagnostic_context.emit(
                            "error",
                            "variable Tensor shape annotations not yet supported",
                            self.tvm_span(shape_annotation.span),
                        )
                        self._diagnostic_context.render()
                    else:
                        # FIXME: use a special node for unknown shape vs no shape?
                        pass  # shape = None
                elif isinstance(shape_annotation, ast.TypeTuple):
                    shape = self.parse_shape(shape_annotation, allow_intro)
                else:
                    self._diagnostic_context.emit(
                        "error",
                        "unsupported shape annotation",
                        self.tvm_span(shape_annotation.span),
                    )
                    self._diagnostic_context.render()

                # parse the dtype annotation
                if isinstance(dtype_annotation, ast.TypeVar) and dtype_annotation.id.name == "_":
                    pass  # dtype = None
                elif isinstance(dtype_annotation, ast.TypeConstant):
                    dtype = dtype_annotation.value  # TODO: parse to TVM DType?
                else:
                    self._diagnostic_context.emit(
                        "error",
                        "Tensor dtype annotations must be concrete or erased",
                        self.tvm_span(dtype_annotation.span),
                    )
                    self._diagnostic_context.render()

                rank = len(shape) if shape is not None else -1
                return (rx.DynTensorType(rank=rank, dtype=dtype, span=span), shape)
            elif ty.id.name == "Tuple":
                field_types = []
                field_shapes = []
                for field in ty.params:
                    fty, fsh = self.transform_type(field, allow_intro=False)
                    field_types.append(fty)
                    field_shapes.append(fsh)
                return relay.TupleType(field_types, self.tvm_span(ty.span)), field_shapes
            # TODO: other types with args, e.g. Ref[T], func types
        self._diagnostic_context.emit("error", "invalid type", span)
        self._diagnostic_context.render()

    def parse_shape(
        self, shape_annotation: Union[ast.TypeTuple, ast.Tuple], allow_intro: bool
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
                    self._diagnostic_context.emit(
                        "error",
                        "non-dimension variables cannot appear in dimension expressions",
                        self.tvm_span(expr.span),
                    )
                    self._diagnostic_context.render()
                return var
            elif allow_intro:
                # introduce TIR variable to scope, e.g. for func params or rx.call_packed
                var = tir.Var(var_name, "int32", self.tvm_span(expr.span))
                self.scope[var_name] = var
                return var
            else:
                self._diagnostic_context.emit(
                    "error",
                    "cannot introduce new dimension variables in this expression",
                    self.tvm_span(expr.span),
                )
                self._diagnostic_context.render()
        elif isinstance(expr, ast.Constant):
            if not isinstance(expr.value, int):
                self._diagnostic_context.emit(
                    "error", "only integer constants are supported", self.tvm_span(expr.span)
                )
                self._diagnostic_context.render()
            return tir.const(expr.value, "int32", self.tvm_span(expr.span))
        else:
            # TODO: parse (simple) PrimExprs
            self._diagnostic_context.emit(
                "error",
                "only dimension variable expressions are currently supported",
                self.tvm_span(expr.span),
            )
            self._diagnostic_context.render()

    # Turns a tuple into an array of PrimExprs
    # Allow arithmetic indicates whether we are letting there be
    # def expr_to_primexpr(self, expr: ast.Expr, allow_arithmetic=False) -> PrimExpr:
    #     if not allow_arithmetic and not isinstance(expr, ast.Var):
    #         # TODO: improve error message
    #         self._diagnostic_context.emit(
    #             "error",
    #             "You can only use single variables here, not an expression",
    #             self.span_to_span(expr.span),
    #         )
    #         self._diagnostic_context.render()
    #     else:
    #         if isinstance(expr, ast.Var):
    #             return tir.Var(expr.id.name, "int32")

    #         # TODO: do all the ops here
    #         elif isinstance(expr, ast.Constant) and isinstance(expr.value, int):
    #             return tir.IntImm("int32", expr.value)
    #         elif isinstance(expr, ast.Call):
    #             if exp.func_name.name == ast.BuiltinOp.Add:
    #                 # TODO: call this fn on args and return primexpr containing result
    #                 assert False
    #             if exp.func_name.name == ast.BuiltinOp.Sub:
    #                 assert False
    #             if exp.func_name.name == ast.BuiltinOp.Mul:
    #                 assert False
    #             if exp.func_name.name == ast.BuiltinOp.Div:
    #                 assert False
    #             if exp.func_name.name == ast.BuiltinOp.Mod:
    #                 assert False
    #         else:
    #             self._diagnostic_context.emit(
    #                 "error",
    #                 "The shape expression can only contain arithmetic operators, integer constants and variables",
    #                 self.tvm_span(expr.span),
    #             )
    #             self._diagnostic_context.render()

    def transform_module(self, mod: ast.Module) -> IRModule:
        for func_name in mod.funcs:
            func = mod.funcs[func_name]
            self.module[func_name] = self.transform_function(func)
        return self.module

    def transform_function(self, func: ast.Function) -> rx.Function:
        with self.new_scope():
            params = []
            for param in func.params:
                ty, shape = self.transform_type(param.ty, allow_intro=True)
                param = self.decl_var(param.name, ty, shape, self.tvm_span(param.span))
                params.append(param)
            new_body = self.transform_block(func.body)
            ret_type, _ = self.transform_type(func.ret_type, allow_intro=False)
        return rx.Function(func.name, params, new_body, ret_type, self.tvm_span(func.span))

    # Stmts:
    # - Assert: probably unsupported for now
    # - Assign: VarBinding
    # - For: ??
    # - If: IfThenElse, must check no empty false branch
    # - Return: just the returned expression, must terminate blocks? (special case if-else)
    # - UnassignedCall: match_shape
    # - With: rx.dataflow
    def transform_stmt(self, stmt: ast.Stmt) -> rx.Expr:
        if isinstance(stmt, ast.Assign):
            if not isinstance(stmt.lhs, ast.Var):
                self._diagnostic_context.emit(
                    "error",
                    "the left hand side of a binding must be a variable",
                    self.tvm_span(stmt.lhs.span),
                )
                self._diagnostic_context.render()
            # TODO: figure out proper way of doing this
            rhs = self.transform_expr(stmt.rhs)
            if isinstance(rhs, relay.Call) and rhs.op == tvm.ir.Op.get("rx.call_packed"):
                allow_intro = True
            else:
                allow_intro = False
            ty, shape = self.transform_type(stmt.ty, allow_intro)
            lhs = self.decl_var(stmt.lhs.id.name, ty, shape, self.tvm_span(stmt.lhs.span))
            return rx.VarBinding(lhs, rhs, self.tvm_span(stmt.span))

        elif isinstance(stmt, ast.If):
            # TODO: proper diagnostics

            # check branches are non-empty
            assert stmt.true.stmts
            assert stmt.false.stmts
            true_assign = stmt.true.stmts[-1]
            false_assign = stmt.false.stmts[-1]

            # check last statement in each branch lines up
            assert isinstance(true_assign, ast.Assign) and isinstance(true_assign.lhs, ast.Var)
            assert isinstance(false_assign, ast.Assign) and isinstance(false_assign.lhs, ast.Var)
            assert true_assign.lhs.id.name == false_assign.lhs.id.name
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
            ite_expr = relay.If(cond, true_branch, false_branch, self.tvm_span(stmt.span))
            var = self.decl_var(var_name, None, None, self.tvm_span(false_assign.span))
            return rx.VarBinding(var, ite_expr, self.tvm_span(stmt.span))

        elif isinstance(stmt, ast.Return):
            return self.transform_expr(stmt.value)

        # match_shape is the ONLY node that doesn't have to be bound to an LHS variable!
        elif isinstance(stmt, ast.UnassignedCall):
            call: synr.ast.Call = stmt.call
            op = self.transform_expr(call.func_name)
            # FIXME: this check is unreachable since transform_expr tries looking up func_name as a
            #        variable and fails
            if op != tvm.ir.Op.get("rx.match_shape"):
                self._diagnostic_context.emit(
                    "error", "the results of calls must be bound or used", self.tvm_span(stmt.span)
                )
                self._diagnostic_context.render()
            if len(stmt.call.params) != 2:
                self._diagnostic_context.emit(
                    "error", "rx.match_shape takes exactly two arguments", self.tvm_span(stmt.span)
                )
                self._diagnostic_context.render()

            lhs = stmt.call.params[0]
            rhs = stmt.call.params[1]

            rhs_expr = self.transform_expr(rhs)
            if not isinstance(lhs, ast.Tuple):
                self._diagnostic_context.emit(
                    "error",
                    "the pattern (lhs) of rx.match_shape must be a tuple",
                    self.tvm_span(lhs.span),
                )
                self._diagnostic_context.render()
            lhs_expr = self.parse_shape(lhs, allow_intro=True)
            return rx.MatchShape(lhs_expr, rhs_expr, self.tvm_span(stmt.span))

        elif isinstance(stmt, ast.With):
            if not isinstance(stmt.rhs, ast.Call):
                self._diagnostic_context.emit(
                    "error", "unsupported with block", self.tvm_span(stmt.span)
                )
                self._diagnostic_context.render()

            call = stmt.rhs
            op = self.transform_expr(call.func_name)

            # TODO: perhaps this ought to be more general

            if op != "rx.dataflow":
                self._diagnostic_context.emit(
                    "error", "unsupported with block type", self.tvm_span(call.span)
                )
                self._diagnostic_context.render()
            if len(call.params) > 0:
                self._diagnostic_context.emit(
                    "error",
                    "dataflow block constructor takes no arguments",
                    self.tvm_span(call.params[0].span),
                )
                self._diagnostic_context.render()
            if len(stmt.lhs) > 0:
                self._diagnostic_context.emit(
                    "error", "dataflow blocks don't bind any patterns", self.tvm_span(stmt.lhs[0].span)
                )
                self._diagnostic_context.render()

            return self.parse_dataflow(stmt.body)

        elif isinstance(stmt, ast.Function):
            func = self.transform_function(stmt)
            func_var = self.decl_var(func.name, None, None, self.tvm_span(stmt.span))
            return rxVarBinding(func_var, func, self.tvm_span(stmt.span))

        else:
            self._diagnostic_context.emit(
                "error",
                "unsupported statement",
                self.tvm_span(stmt.span),
            )
            self._diagnostic_context.render()

    def parse_dataflow(self, block: ast.Block):
        assert len(block.stmts) > 0, "should never have an empty dataflow block"
        bindings = []

        with self.new_scope():
            for binding_stmt in block.stmts[:-1]:
                if not isinstance(binding_stmt, ast.Assign):
                    self._diagnostic_context.emit(
                        "error",
                        "only bindings are supported in dataflow blocks",
                        self.tvm_span(binding_stmt.span),
                    )
                    self._diagnostic_context.render()
                binding = self.transform_stmt(binding_stmt)
                bindings.append(binding)

            output_stmt = block.stmts[-1]
            if not isinstance(output_stmt, ast.Return):
                self._diagnostic_context.emit(
                    "error",
                    "dataflow blocks must end with returning the output variables",
                    self.tvm_span(output_stmt.span),
                )
                self._diagnostic_context.render()

            ret_val = output_stmt.value
            if isinstance(ret_val, ast.Var):
                ret_val = ast.Tuple(values=[ret_val], span=ret_val.span)

            if not isinstance(ret_val, ast.Tuple) or any([not isinstance(f, ast.Var) for f in ret_val.values]):
                self._diagnostic_context.emit(
                    "error",
                    "the returned values must be variables",
                    self.tvm_span(ret_val.span),
                )

            ret_vars = [self.transform_expr(v) for v in ret_val.values]

        # parent scope
        for v in ret_vars:
            self.scope[v.id] = v

        return rxDataflowBlock(bindings, ret_vars, self.tvm_span(block.span))


    # Exprs:
    # - ArrayLiteral: unsupported for now?
    # - Attr: use for .shape, and intrinsic/special operator namespace
    # - Call
    # - Constant
    # - DictLiteral: unsupported for now
    # - Slice: unsupported for now, could desugar to slice op
    # - Tuple
    # - Var
    def transform_expr(self, expr: ast.Expr) -> rxExpr:
        if isinstance(expr, ast.Attr):
            obj = self.transform_expr(expr.object)
            field_name = expr.field.name
            # TODO: use some kind of proper identifier? str bad
            if isinstance(obj, str):
                return obj + "." + field_name
            elif field_name == "shape":
                return rxCall("rx.shape_of", [obj], self.tvm_span(expr.span))
            else:
                self._diagnostic_context.emit(
                    "error", "unsupported attribute", self.tvm_span(expr.span)
                )
                self._diagnostic_context.render()
        if isinstance(expr, ast.Call):
            op = expr.func_name
            if isinstance(op, ast.Var):
                args = []
                for arg in expr.params:
                    args.append(self.transform_expr(arg))
                if op.id.name in self.scope:
                    op = self.transform_expr(op)
                else:
                    # TODO: fix
                    op = op.id.name
                return rxCall(op, args, self.tvm_span(expr.span))
                # if exp.func_name.id.name in self.str_to_var:
                #     return self.str_to_var[exp.func_name.id.name]
                # else:
                #     name = exp.func_name.id.name
                #     relax_fn = getattr(self.definition_scope, name, None)
                #     # builtin operator
                #     if relax_fn is None:
                #         return rxCall(rxGetBuiltin(name), params, None)
                #     else:
                #         self.module[name] = relax_fn.module[name]
                #         # todo: globalvar equality? use global str -> id map?
                #         ident = Id(exp.func_name.id.name)
                #         return rxCall(rxGlobalVar(ident, None, None), params, None)
            elif isinstance(op, ast.Op):
                assert False, "TODO: sugar for python built in operators"
                # if exp.func_name.name == ast.BuiltinOp.Subscript:
                #     tensor = self.transform_expr(exp.params[0])
                #     indicies = []
                #     for index in exp.params[1].values:
                #         indicies.append(self.transform_expr(index))
                #     # TODO: Replace with relax node
                #     return rxTensorSlice(tensor, indicies, None)
                # elif exp.func_name.name == ast.BuiltinOp.Add:
                #     params = []
                #     for arg in exp.params:
                #         params.append(self.transform_expr(arg))
                #     # TODO: Replace with relax node
                #     return rxCall("add", [params[0], params[1]], None)
            else:
                self._diagnostic_context.emit(
                    "error", "unsupported function", self.tvm_span(expr.span)
                )
                self._diagnostic_context.render()
        elif isinstance(expr, ast.Tuple):
            fields = [self.transform_expr(field) for field in expr.values]
            return rxTuple(fields, self.tvm_span(expr.span))
        elif isinstance(expr, ast.Var):
            var_name = expr.id.name
            if var_name == "rx":
                return "rx"
            if var_name not in self.scope:
                self._diagnostic_context.emit(
                    "error", "undefined variable", self.tvm_span(expr.span)
                )
                self._diagnostic_context.render()
            return self.scope[var_name]
        else:
            self._diagnostic_context.emit(
                "error", "unsupported expression", self.tvm_span(expr.span)
            )
            self._diagnostic_context.render()

    def transform_block(self, block: ast.Block) -> rxSeqExpr:
        # a block of statements needs to be converted to a SeqExpr of binding blocks
        blocks = []
        current_block = []
        for stmt in block.stmts[:-1]:
            parsed_stmt = self.transform_stmt(stmt)
            if isinstance(parsed_stmt, rxDataflowBlock):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                blocks.append(parsed_stmt)
            else:
                assert isinstance(parsed_stmt, rxBinding)
                current_block.append(parsed_stmt)
        if len(current_block) > 0:
            blocks.append(current_block)

        ret_stmt = block.stmts[-1]
        if not isinstance(ret_stmt, ast.Return):
            self._diagnostic_context.emit(
                "error",
                "block must end with a returned expression",
                self.tvm_span(ret_stmt.span),
            )
            self._diagnostic_context.render()
        ret_expr = self.transform_stmt(ret_stmt)

        return rxSeqExpr(blocks, ret_expr, self.tvm_span(block.span))

    def transform_parameter(self, expr: ast.Parameter) -> rxExpr:
        pass


class TVMDiagnosticContext(synr.DiagnosticContext):
    def __init__(self, tvm_diag_ctx):
        self.tvm_diag_ctx = tvm_diag_ctx
        self.str_to_source_name = {}

    def add_source(self, name: str, source: str) -> None:
        """Add a file with source code to the context. This will be called
        before any call to :py:func:`emit` that contains a span in this
        file.
        """
        src_name = self.tvm_diag_ctx.module.source_map.add(name, source)
        self.str_to_source_name[name] = src_name

    def emit(self, level: str, message: str, span: tvm.ir.Span) -> None:
        """Called when an error has occured."""

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

    def render(self) -> Optional[Any]:
        """Render out all error messages. Can either return a value or raise
        and execption.
        """
        self.tvm_diag_ctx.render()


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
    ir_module = tvm.IRModule({})
    diag_ctx = diagnostics.DiagnosticContext(ir_module, diagnostics.get_renderer())
    diag_ctx = TVMDiagnosticContext(diag_ctx)
    ast = synr.to_ast(f, diag_ctx)
    definition_scope = inspect.getmodule(f)
    module = RelaxTransformer(definition_scope, diag_ctx).do_transform(ast, diag_ctx)
    return RelaxDecoratedFn(f.__name__, module, diag_ctx)
