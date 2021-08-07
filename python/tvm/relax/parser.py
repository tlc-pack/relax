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
relax_scope = [] # A stack of dictionaries representing the scope
var_table = {}

# Skeleton AST so we can get prototype working before this PR is merged
class rxNode:
    pass

# A node that will desugar into a different AST node in a subsequent pass
class rxFrontendNode:
    def __init__(self, span):
        self.span = span

class rxExpr(rxNode):
    def __init__(self, span):
        self.shape = None
        self.checked_type = None
        self.span = span

class rxVar(rxExpr):

    def __init__(self, id, ty, shape_annotation, span):
        super().__init__(span)
        self.shape_annotation = shape_annotation
        self.type_annotation = ty
        self.id = id

class rxGlobalVar(rxVar):
    def __init__(self, id, span):
        super().__init__(self, id, span)

class rxDataflowVar(rxVar):
    pass

class rxBinding(rxNode):

    def __init__(self, var, rhs, span):
        self.var = var
        self.rhs = rhs
        super().__init__(self, span)

# Allows arbitrary exprs on the left and the right
# Desugars into two rxMatchShapeBinding
# TODO: might be worth not parsing this into its own node..
class rxFrontendMatchShapeExprs(rxFrontendNode):

    def __init__(self, lhs, rhs, span):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(span)

# 
class rxMatchShapeBinding(rxNode):
    def __init__(self, binding, shape, span):
        self.binding = binding # Array[PrimExpr]
        self.shape = shape # Expr (type is shape tuple)
        super().__init__(self, span)

class rxShapeTuple(rxExpr):
    def __init__(self, dims, span):
        self.dims = dims
        super().__init__(span)


class rxFunction(rxExpr):
    def __init__(self, name, args, body, ret_type, span):
        self.name = name
        self.args = args
        self.body = body
        self.ret_type = ret_type
        super().__init__(span)

class rxBlock(rxExpr):
    def __init__(self, body):
        self.body = body

class rxDataflowBlock(rxBlock):
    def __init__(self, body):
        super().__init__(self, body)


class rxIfThenElse(rxExpr):
    def __init__(self, cond, if_true, then_else):
        self.cond = cond
        self.if_true = if_true
        self.then_else = then_else

class rxType:
    def __init__(self, span):
        self.span = span

class rxDim(rxType):
    pass

class rxTensor(rxType):
    def __init__(self, dtype, span):
        self.dtype = dtype
        super().__init__(span)

class rxShapeOf(rxExpr):
    def __init__(self, expr):
        self.expr = expr

class rxLet(rxExpr):
    def __init__(self, bindings, body):
        self.bindings = bindings
        self.body = body

class rxCall(rxExpr):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

class rxGetBuiltin(rxExpr):
    def __init__(self, builtin_name):
        self.builtin_name = builtin_name

class rxTensorSlice(rxExpr):
    def __init__(self, tensor, indices):
        self.tensor = tensor
        self.indices = indices

# TODO: What is this doing?
#expr.Function.__str__ = print_fn # type: ignore

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


    def decl_var(self, name, ty, shape_annotation, span=None):
        var = rxVar(name, ty, shape_annotation, span)
        self.str_to_var[name] = var
        return var

    # Returns type, shape_annotation
    def to_type(self, ty):
        if ty is None:
            return None

        if isinstance(ty, ast.TypeVar):
            if ty.id.name == "Tensor":
                span = self.span_to_span(ty.span)
                return rxTensor(None, span), None
        
        if isinstance(ty, ast.TypeApply):
            if ty.id.name == "Tensor":
                # TODO: add more dtypes, maybe define elsewhere
                allowed_dtypes = ["int32", "float32", "int8", "fp16"]
                dims = []
                dtype = ""
                assert len(ty.params) == 2
                shape_param = ty.params[0]
                dtype_param = ty.params[1]
                # Check whether the Tensor is shape / dtype erased
                shape_erased = False
                dtype_erased = False
                
                if (isinstance(shape_param, ast.TypeVar)) and shape_param.id.name == "_":
                    shape_erased = True
                if (isinstance(dtype_param, ast.TypeVar)) and dtype_param.id.name == "_":
                    dtype_erased = True

                if not shape_erased:
                    if isinstance(shape_param, ast.TypeTuple):
                        for shape_dim in shape_param.values:

                            # TODO: use to_primexpr or whatever
                            if isinstance(shape_dim, ast.Var):
                                dim = rxDim(tir.Var(shape_dim.name))
                            elif isinstance(shape_dim, ast.Constant) and isinstance(shape_dim.value, int):
                                dim = rxDim(tir.IntImm("int32", shape_dim.value))
                            else:
                                self._diagnostic_context.emit('error', "shape annotation must be only vars or consts for now", self.span_to_span(ty.span))            
                                self._diagnostic_context.render()
                            dims.append(dim)
                    else:
                        self._diagnostic_context.emit('error', "Tensor shape must be erased or be a tuple", self.span_to_span(ty.span))    
                        self._diagnostic_context.render()     
                if not dtype_erased:
                    if dtype_param.value in allowed_dtypes:
                        dtype = dtype_param.value
                    else:
                        self._diagnostic_context.emit('error', "dtype must be erased or one of " + str(allowed_dtypes), self.span_to_span(ty.span))            
                        self._diagnostic_context.render()
                
                return rxTensor(dtype, None), dims

        self._diagnostic_context.emit('error', "invalid type", self.span_to_span(ty.span))

    # Turns a tuple into an array of PrimExprs
    # Allow arithmetic indicates whether we are letting there be 
    def expr_to_primexpr(self, expr: ast.Expr, allow_arithmetic=False) -> PrimExpr:
        if not allow_arithmetic and not isinstance(expr, ast.Var):
            #TODO: improve error message
            self._diagnostic_context.emit('error', "You can only use single variables here, not an expression", self.span_to_span(expr.span))
            self._diagnostic_context.render()
        else:
            if isinstance(expr, ast.Var):
                return tir.Var(expr.id.name, "int32")
                
            # TODO: do all the ops here
            elif isinstance(expr, ast.Constant) and isinstance(expr.value, int):
                return tir.IntImm("int32", expr.value)
            elif isinstance(expr, ast.Call):
                if exp.func_name.name == ast.BuiltinOp.Add:
                    # TODO: call this fn on args and return primexpr containing result
                    assert False
                if exp.func_name.name == ast.BuiltinOp.Sub:
                    assert False
                if exp.func_name.name == ast.BuiltinOp.Mul:
                    assert False
                if exp.func_name.name == ast.BuiltinOp.Div:
                    assert False
                if exp.func_name.name == ast.BuiltinOp.Mod:
                    assert False
            else:
                self._diagnostic_context.emit('error', "The shape expression can only contain arithmetic operators, integer constants and variables", self.span_to_span(expr.span))
                self._diagnostic_context.render()


    def transform_module(self, mod: ast.Module) -> IRModule:
        for func_name in mod.funcs:
            func = mod.funcs[func_name]
            self.module[func_name] = self.transform_function(func)
        return self.module

    def transform_function(self, func: ast.Function) -> rxFunction:
        params = []
        for param in func.params:
            ty, shape_dims = self.to_type(param.ty)
            param = self.decl_var(param.name, ty, None)
            params.append(param)
        new_body = self.transform_block(func.body)
        ret_type = self.to_type(func.ret_type)
        return rxFunction(func.name, params, new_body, ret_type, None)

    def transform_stmt(self, stmt: ast.Stmt) -> relax.Expr:
        if isinstance(stmt, ast.Assign):
            assert isinstance(stmt.lhs, ast.Var)
            lhs = self.decl_var(stmt.lhs.id.name, None, None)
            rhs = self.transform_expr(stmt.rhs)
            # TODO: Replace with relax node
            self.blocks[-1].append(rxBinding(lhs, rhs))
            return None
        elif isinstance(stmt, ast.Return):
            return self.transform_expr(stmt.value)
        # match_shape is the ONLY node that doesn't have to be bound to an LHS variable!
        elif (isinstance(stmt, ast.UnassignedCall) and isinstance(stmt.call.func_name, ast.Var)
            and stmt.call.func_name.id.name == "match_shape"):
            if len(stmt.call.params) != 2:
                self._diagnostic_context.emit('error', "match_shape takes exactly two arguments", self.span_to_span(stmt.span))
                self._diagnostic_context.render()

            lhs = stmt.call.params[0]
            rhs = stmt.call.params[1]
            
            # If RHS is a tuple, turn it into a ShapeTuple, otherwise, process normally
            if isinstance(rhs, ast.Tuple):
                arithmetic_primexprs = []
                for elem in rhs.values:
                    arithmetic_primexprs.append(self.expr_to_primexpr(elem, allow_arithmetic=True))
                rhs_expr = rxShapeTuple(arithmetic_primexprs, self.span_to_span(rhs.span))
            else:
                rhs_expr = self.transform_expr(rhs)
            
            # If LHS is a tuple of variables, then we use the binding match shape
            # If it is an Expr, we use the sugared match_shape (and will insert bindings later)
            if isinstance(lhs, ast.Tuple):
                binding_tir_vars = []
                for elem in lhs.values:
                    # Here we only are defining variables so we don't allow arithmetic expressions
                    binding_tir_vars.append(self.expr_to_primexpr(elem))
                self.blocks[-1].append(rxMatchShapeBinding(binding_tir_vars, rhs_expr))
            else:
                lhs_expr = self.transform_expr(lhs)
                self.blocks[-1].append(rxFrontendMatchShapeExprs(lhs_expr, rhs_expr, stmt.span))
        else:
            self._diagnostic_context.emit('error', "only variable left-hand sides are supported in Relay", self.span_to_span(stmt.span))
            self._diagnostic_context.render()

    def transform_expr(self, exp: ast.Expr) -> rxExpr:
        if isinstance(exp, ast.Call):
            if isinstance(exp.func_name, ast.Var):
                params = []
                for arg in exp.params:
                    params.append(self.transform_expr(arg))

                if exp.func_name.id.name in self.str_to_var:
                    return self.str_to_var[exp.func_name.id.name]
                else:
                    name = exp.func_name.id.name
                    relax_fn = getattr(self.definition_scope, name, None)
                    # builtin operator
                    if relax_fn is None:
                        return rxCall(rxGetBuiltin(name), params, None)
                    else:
                        self.module[name] = relax_fn.module[name]
                        # todo: globalvar equality? use global str -> id map?
                        ident = Id(exp.func_name.id.name)
                        return rxCall(rxGlobalVar(ident, None, None), params, None)

            elif isinstance(exp.func_name, ast.Op):
                if exp.func_name.name == ast.BuiltinOp.Subscript:
                    tensor = self.transform_expr(exp.params[0])
                    indicies = []
                    for index in exp.params[1].values:
                        indicies.append(self.transform_expr(index))
                    # TODO: Replace with relax node
                    return rxTensorSlice(tensor, indicies, None)
                elif exp.func_name.name == ast.BuiltinOp.Add:
                    params = []
                    for arg in exp.params:
                        params.append(self.transform_expr(arg))
                    # TODO: Replace with relax node
                    return rxCall("add", [params[0], params[1]], None)

            self._diagnostic_context.emit('error', "unsupported function", self.span_to_span(exp.span))
            self._diagnostic_context.render()
        elif isinstance(exp, ast.Attr):
            field_name = exp.field.name
            tensor = self.transform_expr(exp.object)

            if field_name == "shape":
                return rxShapeOf(tensor)
            else:
                self._diagnostic_context.emit('error', "unsupported function", self.span_to_span(exp.span))
                self._diagnostic_context.render()
        elif isinstance(exp, ast.Function):
            return self.transform_function(exp)
        elif isinstance(exp, ast.Tuple):
            assert False
        elif isinstance(exp, ast.Var):
            return self.str_to_var[exp.id.name]
        else:
            self._diagnostic_context.emit('error', f"don't support this construct {type(exp)}", self.span_to_span(exp.span))
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
        return rxLet(bindings, ret_expr)

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

def relax(f):
    ir_module = tvm.IRModule({})
    diag_ctx = diagnostics.DiagnosticContext(ir_module, diagnostics.get_renderer())
    diag_ctx = TVMDiagnosticContext(diag_ctx)
    ast = synr.to_ast(f, diag_ctx)
    definition_scope = inspect.getmodule(f)
    module = RelaxTransformer(definition_scope, diag_ctx).do_transform(ast, diag_ctx)
    return RelaxDecoratedFn(f.__name__, module, diag_ctx)

@relax
def my_test(x: Tensor[_, "float32"]):
    match_shape(x.shape, (1, 2, 3))

print(my_test.module['my_test'])