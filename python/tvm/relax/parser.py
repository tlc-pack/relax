from __future__ import annotations

import inspect
from typing import TypeVar, Generic, Union, Dict
from io import StringIO

import tvm
from tvm.ir.module import IRModule
from tvm.relay.base import Id
from tvm.ir import diagnostics
from tvm import tir, relax

import numpy as np

import synr
from synr import ast, Transformer
from synr.diagnostic_context import DiagnosticContext
from tvm.relay.op.strategy.generic import conv1d_strategy


# TODO: make this better
var_table = {}

# Skeleton AST so we can get prototype working before this PR is merged
class rxNode:
    pass

class rxExpr(rxNode):
    def __init__(self):
        self.shape = None
        self.checked_type = None

class rxVar(rxExpr):

    def __init__(self, name):
        super.__init__(self)
        self.shape_annotation = None
        self.type_annotation = None
        if name not in var_table:
            self.id = name
            var_table.add(name)
        else:
            assert False, "All variable names must be unique, name is: " + name

class rxDataflowVar(rxVar):
    pass

class rxBinding(rxNode):

    def __init__(self, var, rhs):
        self.var = var
        self.rhs = rhs

class rxMatchShape(rxNode):

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

# TODO: is dim a tir var or any algebraic PrimExpr?
class Dim:
    def __init__(self, name):
        self.name = name

class ShapeTuple(rxExpr):
    def __init__(self, dims):
        self.dims = dims

class rxFunction(rxExpr):
    def __init__(self, args, body):
        self.args = args
        self.body = body

class rxBlock(rxExpr):
    def __init__(self, body):
        self.body = body

class rxDataflowBlock(rxBlock):
    def __init__(self, body):
        super.__init__(self, body)

class rxBasicBlock(rxBlock):
    def __init__(self, body):
        super.__init__()

class rxIfThenElse(rxExpr):
    def __init__(self, cond, if_true, then_else):
        self.cond = cond
        self.if_true = if_true
        self.then_else = then_else





# TODO: What is this doing?
expr.Function.__str__ = print_fn # type: ignore

# TODO: Replace with a real pretty print method once we have the real AST
def pretty_print(f):
    print(f)

class RelaxTransformer(Transformer):
    def __init__(self, definition_scope, diag_ctx):
        self.definition_scope = definition_scope
        self.diag_ctx = diag_ctx
        self.str_to_var = {}
        self.blocks = []
        self.module = {}
        super().__init__()

    def span_to_span(self, span: synr.Span) -> tvm.ir.Span:
        src_name = self.diag_ctx.str_to_source_name[span.filename]
        tvm_span = tvm.ir.Span(src_name, span.start_line, span.end_line, span.start_column, span.end_column)
        return tvm_span


    def decl_var(self, name, ty, span=None):
        identifier = Id(name)
        # TODO: Replace with relax node
        var = expr.Var(identifier, ty, span)
        self.str_to_var[name] = var
        return var

    def to_type(self, ty):
        if ty is None:
            return None

        if isinstance(ty, ast.TypeVar):
            if ty.id.name == "Tensor":
                span = self.span_to_span(ty.span)
                # TODO: Replace with relax node
                return expr.Tensor(None, None, span)

        if isinstance(ty, ast.TypeApply):
            if ty.id.name == "Tensor":
                dims = []
                # TODO(@jroesch): add support for dtype
                for param in ty.params:
                    if isinstance(param, ast.TypeConstant):
                        # TODO: Replace with relax node
                        dim = expr.TIRExpr(tir.IntImm("int32", param.value), None)
                        dims.append(dim)
                # TODO: Replace with relax node
                return expr.Tensor(expr.Tuple(dims, span=None), None, None)

        self._diagnostic_context.emit('error', "invalid type", self.span_to_span(ty.span))
        self._diagnostic_context.render()

    def transform_module(self, mod: ast.Module) -> IRModule:
        for func_name in mod.funcs:
            func = mod.funcs[func_name]
            self.module[func_name] = self.transform_function(func)
        return self.module

    def transform_function(self, func: ast.Function) -> relax.Function: # TODO: update once relax ast finalized
        params = []
        for param in func.params:
            ty = self.to_type(param.ty)
            param = self.decl_var(param.name, ty, None)
            params.append(param)
        new_body = self.transform_block(func.body)
        ret_type = self.to_type(func.ret_type)
        print(new_body)
        # TODO: Replace with relax node
        return expr.Function(func.name, params, new_body, ret_type, None)

    def transform_stmt(self, stmt: ast.Stmt) -> relax.Expr:
        if isinstance(stmt, ast.Assign):
            assert isinstance(stmt.lhs, ast.Var)
            lhs = self.decl_var(stmt.lhs.id.name, None, None)
            rhs = self.transform_expr(stmt.rhs)
            # TODO: Replace with relax node
            self.blocks[-1].append(expr.Binding(lhs, rhs))
            return None
        elif isinstance(stmt, ast.Return):
            return self.transform_expr(stmt.value)
        else:
            self._diagnostic_context.emit('error', "only variable left-hand sides are supported in Relay", stmt.span)
            self._diagnostic_context.render()

    def transform_expr(self, exp: ast.Expr) -> relax.Expr: # TODO: update once we have real relax AST
        if isinstance(exp, ast.Call):
            if isinstance(exp.func_name, ast.Var):
                params = []
                for arg in exp.params:
                    params.append(self.transform_expr(arg))

                if exp.func_name.id.name == "broadcast_shape":
                    if len(params) != 2:
                        self._diagnostic_context.emit('error', f"broadcast_shape only takes 2 arguments {params.len()}", exp.span)
                        self._diagnostic_context.render()
                    # TODO: Replace with relax node
                    return expr.BroadcastShape(params[0], params[1], span=None)
                elif exp.func_name.id.name == "compute":
                    if len(params) != 2:
                        self._diagnostic_context.emit('error', f"compute only takes 2 arguments {params.len()}", exp.span)
                        self._diagnostic_context.render()
                    # TODO: Replace with relax node
                    return expr.Compute(params[0], params[1], span=None)
                else:
                    if exp.func_name.id.name in self.str_to_var:
                        return self.str_to_var[exp.func_name.id.name]
                    else:
                        name = exp.func_name.id.name
                        relax_fn = getattr(self.definition_scope, name, None)
                        # builtin operator
                        if relax_fn is None:
                            # TODO: Replace with relax node
                            return expr.Call(op.Op.get(name), params, None)
                        else:
                            self.module[name] = relax_fn.module[name]
                            # todo: globalvar equality? use global str -> id map?
                            ident = Id(exp.func_name.id.name)
                            # TODO: Replace with relax node
                            return expr.Call(expr.GlobalVar(ident, None, None), params, None)
                    # TODO: Where is this supposed to be?? 
                    self._diagnostic_context.emit('error', f"unknown functionc all {len(params)}", exp.span)
                    self._diagnostic_context.render()
            elif isinstance(exp.func_name, ast.Op):
                if exp.func_name.name == ast.BuiltinOp.Subscript:
                    tensor = self.transform_expr(exp.params[0])
                    indicies = []
                    for index in exp.params[1].values:
                        indicies.append(self.transform_expr(index))
                    # TODO: Replace with relax node
                    return expr.TensorSlice(tensor, indicies, None)
                elif exp.func_name.name == ast.BuiltinOp.Add:
                    params = []
                    for arg in exp.params:
                        params.append(self.transform_expr(arg))
                    # TODO: Replace with relax node
                    return expr.Add(params[0], params[1], None)

            self._diagnostic_context.emit('error', "unsupported function", exp.span)
            self._diagnostic_context.render()
        elif isinstance(exp, ast.Attr):
            field_name = exp.field.name
            tensor = self.transform_expr(exp.object)

            if field_name == "shape":
                # TODO: Replace with relax node
                return expr.ShapeOf(tensor, None)
            else:
                self._diagnostic_context.emit('error', "unsupported function", exp.span)
                self._diagnostic_context.render()
        elif isinstance(exp, ast.Function):
            print(exp)
            return self.transform_function(exp)
        elif isinstance(exp, ast.Tuple):
            assert False
        elif isinstance(exp, ast.Var):
            return self.str_to_var[exp.id.name]
        else:
            self._diagnostic_context.emit('error', f"don't support this construct {type(exp)}", exp.span)
            self._diagnostic_context.render()

    def enter_block(self):
        self.blocks.append([])

    def exit_block(self):
        back = self.blocks[-1]
        self.blocks.pop()
        return back

    def transform_block(self, block: ast.Block) -> relax.expr:
        self.enter_block()

        for stmt in block.stmts[:-1]:
            assert self.transform_stmt(stmt) is None

        ret_expr = self.transform_stmt(block.stmts[-1])
        # assert ret_expr is not None

        bindings = self.exit_block()
        # TODO: Replace with relax node
        return expr.Let(bindings, ret_expr, span=None)

    def transform_parameter(self, expr: ast.Parameter) -> relax.Expr:
        pass

    def transform_type(self, ty: ast.Type) -> relax.Type:
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
        assert isinstance(span, tvm.ir.span), "Expected tvm.ir.span, but got " + str(type(span))

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

def relax(f):
    ir_module = tvm.IRModule({})
    diag_ctx = diagnostics.DiagnosticContext(ir_module, diagnostics.get_renderer())
    diag_ctx = TVMDiagnosticContext(diag_ctx)
    ast = synr.to_ast(f, diag_ctx)
    definition_scope = inspect.getmodule(f)
    # Why have diag context at transform time? TK?
    # TODO: Replace RelaxTransformer with new transformation 
    module = RelaxTransformer(definition_scope, diag_ctx).do_transform(ast, diag_ctx)
    return RelaxDecoratedFn(f.__name__, module, diag_ctx)