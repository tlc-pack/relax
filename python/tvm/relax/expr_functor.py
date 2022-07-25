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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, arguments-differ
"""The expression functor of Relax."""
import tvm
from typing import Optional
from tvm.ir import Op
from tvm.ir.base import structural_equal
from tvm.ir.module import IRModule
from .ty import DynTensorType
from .expr import Type, Span, Expr
from .expr import Function, ExternFunc
from .expr import Constant, Var, DataflowVar
from .expr import ShapeExpr, RuntimeDepShape
from .expr import GlobalVar, SeqExpr, Tuple
from .expr import Call, If, TupleGetItem
from .expr import Binding, MatchShape, VarBinding
from .expr import BindingBlock, DataflowBlock
from .expr import _update_shape, _update_type
from .block_builder import BlockBuilder
from . import _ffi_api

expr_names = [
    "constant",
    "tuple",
    "var",
    "dataflow_var",
    "shape_expr",
    "runtime_dep_shape",
    "extern_func",
    "global_var",
    "function",
    "call",
    "seq_expr",
    "if",
    "op",
    "tuple_getitem",
]
other_names = [
    "var_binding",
    "match_shape",
    "binding_block",
    "dataflow_block",
    "var_def",
    "dataflow_var_def",
]
visit_func_names = [
    "visit_expr",
    "visit_binding",
    "visit_binding_block",
    "visit_var_def",
    "visit_type",
    "visit_span",
] + [f"visit_{name}_" for name in (expr_names + other_names)]
post_order_func_names = [f"rewrite_{name}_post_order" for name in expr_names]

# @tvm._ffi.register_object("relax.ExprFunctor")
class ExprFunctor(tvm.runtime.Object):
    """TODO"""


# @tvm._ffi.register_object("relax.ExprVisitor")
class ExprVisitor(ExprFunctor):
    """TODO"""


# @tvm._ffi.register_object("relax.ExprMutatorBase")
class ExprMutatorBase:
    """TODO"""


# @tvm._ffi.register_object("relax.ExprMutator")
class ExprMutator:
    """TODO"""


@tvm._ffi.register_object("expr_functor.PyExprVisitor")
class _PyExprVisitor(tvm.runtime.Object):
    """TODO"""


class PyExprVisitor(_PyExprVisitor):
    def visit_expr(self, expr: Expr) -> None:
        return _ffi_api.PyExprVisitorVisitExpr(self._outer(), expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprVisitorVisitBinding(self._outer(), binding)

    def visit_binding_block(self, block: BindingBlock) -> None:
        return _ffi_api.PyExprVisitorVisitBindingBlock(self._outer(), block)

    def visit_var_def(self, var: Var) -> None:
        return _ffi_api.PyExprVisitorVisitVarDef(self._outer(), var)


@tvm._ffi.register_object("expr_functor.PyExprMutator")
class _PyExprMutator(tvm.runtime.Object):
    """TODO"""


class PyExprMutator(_PyExprMutator):
    def visit_expr(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitExpr(self._outer(), expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprMutatorVisitBinding(self._outer(), binding)

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        return _ffi_api.PyExprMutatorVisitBindingBlock(self._outer(), block)

    def visit_var_def(self, var: Var) -> Var:
        return _ffi_api.PyExprMutatorVisitVarDef(self._outer(), var)


def visitor(visitor_cls=None):
    import functools
    import weakref

    def _extract(inst, name):
        def method(*args, **kwargs):
            return getattr(inst, name)(*args, **kwargs)

        return method

    class PyVisitor(_PyExprVisitor):
        def __init__(self, *args, **kwargs) -> None:
            "Constructor."
            self.handle = None
            self._inst = visitor_cls(*args, **kwargs)

            packed_value = []

            for func_name in visit_func_names:
                if hasattr(self._inst, func_name):
                    if hasattr(PyExprVisitor, func_name) and (
                        getattr(visitor_cls, func_name) == getattr(PyExprVisitor, func_name)
                    ):
                        continue
                    packed_value.append(func_name)
                    packed_value.append(_extract(self._inst, func_name))

            self.__init_handle_by_constructor__(_ffi_api.MakeExprVisitor, *packed_value)

            self._inst._outer = weakref.ref(self)

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

        def __setattr__(self, name, value):
            """TODO"""
            if name not in ["_inst", "key", "handle"]:
                self._inst.__setattr__(name, value)
            else:
                super(PyVisitor, self).__setattr__(name, value)

    functools.update_wrapper(PyVisitor.__init__, visitor_cls.__init__)
    PyVisitor.__name__ = visitor_cls.__name__
    PyVisitor.__doc__ = visitor_cls.__doc__
    PyVisitor.__module__ = visitor_cls.__module__
    return PyVisitor


def mutator(mutator_cls=None):
    import functools
    import weakref

    def _extract(inst, name):
        def method(*args, **kwargs):
            return getattr(inst, name)(*args, **kwargs)

        return method

    class PyMutator(_PyExprMutator):
        def __init__(self, *args, **kwargs) -> None:
            "Constructor."
            self.handle = None
            self._inst = mutator_cls(*args, **kwargs)

            packed_value = []

            for func_name in visit_func_names:
                if hasattr(self._inst, func_name):
                    if hasattr(PyExprMutator, func_name) and (
                        getattr(mutator_cls, func_name) == getattr(PyExprMutator, func_name)
                    ):
                        continue
                    packed_value.append(func_name)
                    packed_value.append(_extract(self._inst, func_name))

            for func_name in post_order_func_names:
                # rewrite_{name}_post_order
                if hasattr(self._inst, func_name):
                    if f"visit_{'_'.join(func_name.split('_')[1:-2])}_" in packed_value:
                        raise RuntimeError(func_name)  # TODO: RuntimeError?
                    packed_value.append(func_name)
                    packed_value.append(_extract(self._inst, func_name))

            self.__init_handle_by_constructor__(_ffi_api.MakeExprMutator, *packed_value)

            self._inst._outer = weakref.ref(self)

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

        def __setattr__(self, name, value):
            """TODO"""
            if name not in ["_inst", "key", "handle"]:  # TODO: Why do we need key here
                self._inst.__setattr__(name, value)
            else:
                super(PyMutator, self).__setattr__(name, value)

    functools.update_wrapper(PyMutator.__init__, mutator_cls.__init__)
    PyMutator.__name__ = mutator_cls.__name__
    PyMutator.__doc__ = mutator_cls.__doc__
    PyMutator.__module__ = mutator_cls.__module__
    return PyMutator


# class ExprFunctor:
#     """
#     An abstract visitor defined over Expr.

#     Defines the default dispatch over expressions, and
#     implements memoization.
#     """

#     def visit_expr(self, expr):
#         """Apply the visitor to an expression."""
#         if isinstance(expr, Constant):
#             ret = self.visit_constant_(expr)
#         elif isinstance(expr, Tuple):
#             ret = self.visit_tuple_(expr)
#         elif isinstance(expr, DataflowVar):
#             ret = self.visit_dataflow_var_(expr)
#         elif isinstance(expr, Var):
#             ret = self.visit_var_(expr)
#         elif isinstance(expr, ShapeExpr):
#             ret = self.visit_shape_expr_(expr)
#         elif isinstance(expr, RuntimeDepShape):
#             ret = self.visit_runtime_dep_shape_(expr)
#         elif isinstance(expr, ExternFunc):
#             ret = self.visit_extern_func_(expr)
#         elif isinstance(expr, GlobalVar):
#             ret = self.visit_global_var_(expr)
#         elif isinstance(expr, Function):
#             ret = self.visit_function_(expr)
#         elif isinstance(expr, Call):
#             ret = self.visit_call_(expr)
#         elif isinstance(expr, SeqExpr):
#             ret = self.visit_seq_expr_(expr)
#         elif isinstance(expr, If):
#             ret = self.visit_if_(expr)
#         elif isinstance(expr, Op):
#             ret = self.visit_op_(expr)
#         elif isinstance(expr, TupleGetItem):
#             ret = self.visit_tuple_getitem_(expr)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(expr)))

#         return ret

#     def visit_constant_(self, op: Constant):
#         raise NotImplementedError()

#     def visit_tuple_(self, op: Tuple):
#         raise NotImplementedError()

#     def visit_dataflow_var_(self, op: DataflowVar):
#         raise NotImplementedError()

#     def visit_var_(self, op: Var):
#         raise NotImplementedError()

#     def visit_shape_expr_(self, op: ShapeExpr):
#         raise NotImplementedError()

#     def visit_runtime_dep_shape_(self, op: RuntimeDepShape):
#         raise NotImplementedError()

#     def visit_extern_func_(self, op: ExternFunc):
#         raise NotImplementedError()

#     def visit_global_var_(self, op: GlobalVar):
#         raise NotImplementedError()

#     def visit_function_(self, op: Function):
#         raise NotImplementedError()

#     def visit_call_(self, op: Call):
#         raise NotImplementedError()

#     def visit_seq_expr_(self, op: SeqExpr):
#         raise NotImplementedError()

#     def visit_if_(self, op: If):
#         raise NotImplementedError()

#     def visit_op_(self, op: Op):
#         raise NotImplementedError()

#     def visit_tuple_getitem_(self, op: TupleGetItem):
#         raise NotImplementedError()


# class ExprVisitor(ExprFunctor):
#     """
#     A visitor over Expr.

#     The default behavior recursively traverses the AST.
#     """

#     def visit_expr(self, expr: Expr) -> None:
#         ExprFunctor.visit_expr(self, expr)

#     def visit_constant_(self, op: Constant) -> None:
#         self.visit_span(op.span)

#         if op.shape_:
#             self.visit_expr(op.shape_)

#     def visit_global_var_(self, op: GlobalVar) -> None:
#         self.visit_span(op.span)

#     def visit_tuple_(self, op: Tuple) -> None:
#         self.visit_span(op.span)
#         for field in op.fields:
#             self.visit_expr(field)

#         if op.shape_:
#             self.visit_expr(op.shape_)

#     def visit_var_(self, op: Var) -> None:
#         self.visit_span(op.span)

#     def visit_dataflow_var_(self, op: DataflowVar) -> None:
#         self.visit_span(op.span)

#     def visit_function_(self, op: Function) -> None:
#         self.visit_span(op.span)
#         for param in op.params:
#             self.visit_var_def(param)

#         self.visit_expr(op.body)

#     def visit_call_(self, op: Call) -> None:
#         self.visit_span(op.span)
#         self.visit_expr(op.op)

#         for ty_arg in op.type_args:
#             self.visit_type(ty_arg)

#         for arg in op.args:
#             self.visit_expr(arg)

#         if op.shape_:
#             self.visit_expr(op.shape_)

#     def visit_if_(self, op: If) -> None:
#         self.visit_span(op.span)
#         self.visit_expr(op.cond)
#         self.visit_expr(op.true_branch)
#         self.visit_expr(op.false_branch)

#     def visit_op_(self, op: Op) -> None:
#         pass

#     def visit_tuple_getitem_(self, op: TupleGetItem) -> None:
#         self.visit_span(op.span)
#         self.visit_expr(op.tuple_value)

#     def visit_shape_expr_(self, op: ShapeExpr) -> None:
#         self.visit_span(op.span)

#     def visit_runtime_dep_shape_(self, op: RuntimeDepShape) -> None:
#         self.visit_span(op.span)

#     def visit_extern_func_(self, op: ExternFunc) -> None:
#         self.visit_span(op.span)

#     def visit_seq_expr_(self, op: SeqExpr) -> None:
#         self.visit_span(op.span)
#         for block in op.blocks:
#             self.visit_binding_block(block)
#         self.visit_expr(op.body)

#     def visit_type(self, t: Type) -> None:
#         pass

#     def visit_span(self, span: Span) -> None:
#         pass

#     def visit_var_binding_(self, binding: VarBinding) -> None:
#         self.visit_expr(binding.value)
#         self.visit_var_def(binding.var)

#     def visit_match_shape_(self, binding: MatchShape) -> None:
#         self.visit_expr(binding.value)
#         self.visit_expr(ShapeExpr(binding.pattern))
#         if binding.var:
#             self.visit_var_def(binding.var)

#     def visit_binding_block_(self, block: BindingBlock) -> None:
#         for binding in block.bindings:
#             self.visit_binding(binding)

#     def visit_dataflow_block_(self, block: DataflowBlock) -> None:
#         for binding in block.bindings:
#             self.visit_binding(binding)

#     def visit_var_def_(self, var: Var) -> None:
#         self.visit_span(var.span)

#         if var.shape_:
#             self.visit_expr(var.shape_)

#     def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
#         self.visit_span(var.span)

#         if var.shape_:
#             self.visit_expr(var.shape_)

#     def visit_binding(self, binding: Binding) -> None:
#         if isinstance(binding, MatchShape):
#             self.visit_match_shape_(binding)
#         elif isinstance(binding, VarBinding):
#             self.visit_var_binding_(binding)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(binding)))

#     def visit_binding_block(self, block: BindingBlock) -> None:
#         if isinstance(block, DataflowBlock):
#             self.visit_dataflow_block_(block)
#         elif isinstance(block, BindingBlock):
#             self.visit_binding_block_(block)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(block)))

#     def visit_var_def(self, var: Var):
#         if isinstance(var, DataflowVar):
#             self.visit_dataflow_var_def_(var)
#         elif isinstance(var, Var):
#             self.visit_var_def_(var)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(var)))


# class ExprMutatorBase(ExprFunctor):
#     """
#     A mutator works in unnormalized form.

#     ExprMutatorBase expects input AST to be in the unnormalized form,
#     i.e., _checked_type_ and shape_ of expressions can be None,
#     and the expressions may nest (and as a result the AST is not in ANF).
#     """

#     def visit_expr(self, expr: Expr) -> Expr:
#         return ExprFunctor.visit_expr(self, expr)

#     def visit_constant_(self, op: Constant) -> Expr:
#         return op

#     def visit_global_var_(self, op: GlobalVar) -> Expr:
#         return op

#     def visit_tuple_(self, op: Tuple) -> Expr:
#         unchanged = True
#         fields = []
#         for field in op.fields:
#             new_field = self.visit_expr(field)
#             fields.append(new_field)
#             unchanged &= field.same_as(new_field)

#         if unchanged:
#             return op
#         else:
#             return Tuple(fields, op.span)

#     def visit_var_(self, op: Var) -> Expr:
#         return op

#     def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
#         return op

#     def visit_function_(self, op: Function) -> Expr:
#         body = self.visit_expr(op.body)

#         if op.body.same_as(body):
#             return op
#         else:
#             return Function(op.params, body, op.ret_type, op.attrs, op.span)

#     def visit_call_(self, call_node: Call) -> Expr:
#         new_op = self.visit_expr(call_node.op)
#         unchanged = call_node.op.same_as(new_op)

#         ty_args = []
#         for ty_arg in call_node.type_args:
#             new_ty_arg = self.visit_type(ty_arg)
#             ty_args.append(new_ty_arg)
#             unchanged &= ty_arg.same_as(new_ty_arg)

#         call_args = []
#         for arg in call_node.args:
#             new_arg = self.visit_expr(arg)
#             call_args.append(new_arg)
#             unchanged &= arg.same_as(new_arg)

#         if unchanged:
#             return call_node
#         else:
#             return Call(new_op, call_args, call_node.attrs, ty_args, call_node.span)

#     def visit_if_(self, op: If) -> Expr:
#         guard = self.visit_expr(op.cond)
#         true_b = self.visit_expr(op.true_branch)
#         false_b = self.visit_expr(op.false_branch)
#         if (
#             op.cond.same_as(guard)
#             and op.true_branch.same_as(true_b)
#             and op.false_branch.same_as(false_b)
#         ):
#             return op
#         else:
#             return If(guard, true_b, false_b, op.span)

#     def visit_op_(self, op: Op) -> Expr:
#         return op

#     def visit_tuple_getitem_(self, op: TupleGetItem) -> Expr:
#         t = self.visit_expr(op.tuple_value)
#         if op.tuple_value.same_as(t):
#             return op
#         else:
#             return TupleGetItem(t, op.index)

#     def visit_shape_expr_(self, op: ShapeExpr) -> Expr:
#         return op

#     def visit_runtime_dep_shape_(self, op: RuntimeDepShape) -> Expr:
#         return op

#     def visit_extern_func_(self, op: ExternFunc) -> Expr:
#         return op

#     def visit_seq_expr_(self, op: SeqExpr) -> Expr:
#         all_blocks_unchanged = True
#         blocks = []
#         for block in op.blocks:
#             new_block = self.visit_binding_block(block)
#             if new_block.bindings:
#                 blocks.append(new_block)
#             all_blocks_unchanged &= block.same_as(new_block)

#         body = self.visit_expr(op.body)
#         if all_blocks_unchanged and op.body.same_as(body):
#             return op
#         else:
#             return SeqExpr(blocks, body, op.span)

#     def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
#         """Mutate BindingBlock.

#         Parameters
#         ----------
#         block: BindingBlock
#             The binding block to be visited.

#         Returns
#         -------
#         block: BindingBlock
#             The binding block after transformation.
#         """
#         bindings = []
#         if isinstance(block, BindingBlock):
#             for binding in block.bindings:
#                 if isinstance(binding, VarBinding):
#                     new_value = self.visit_expr(binding.value)
#                     bindings.append(VarBinding(binding.var, new_value, binding.span))
#                 elif isinstance(binding, MatchShape):
#                     new_value = self.visit_expr(binding.value)
#                     bindings.append(
#                         MatchShape(new_value, binding.pattern, binding.var, binding.span)
#                     )
#                 else:
#                     raise TypeError("Invalid type: {0}".format(type(block)))
#         else:
#             raise TypeError("Invalid type: {0}".format(type(block)))
#         if isinstance(block, DataflowBlock):
#             return DataflowBlock(bindings)
#         else:
#             return BindingBlock(bindings)

#     def visit_type(self, t: Type) -> Type:
#         return t


# class ExprMutator(ExprMutatorBase):
#     """
#     A mutator works in normal form.

#     ExprMutator expects input AST to be in the normal form, i.e., the expressions are normalized(no
#     nesting and hence the AST is in ANF), and all checked_type_ and shape_ of expressions are
#     available. Note: We can use relax.transform.Normalize()(mod) to transform relax IR into
#     the normal form.
#     """

#     def __init__(self, mod: Optional[IRModule] = None) -> None:
#         super().__init__()
#         self.builder_ = BlockBuilder(mod)
#         self.var_remap_ = dict()

#     def visit_expr(self, expr) -> Expr:
#         return self.builder_.normalize(ExprFunctor.visit_expr(self, expr))

#     def visit_tuple_(self, op: Tuple) -> Expr:
#         unchanged = True
#         fields = []
#         for field in op.fields:
#             new_field = self.visit_expr(field)
#             fields.append(new_field)
#             unchanged &= field.same_as(new_field)

#         if unchanged:
#             return op
#         else:
#             new_tuple = Tuple(fields, op.span)
#             return new_tuple

#     def visit_var_(self, op: Var) -> Expr:
#         if op.vid in self.var_remap_:
#             return self.var_remap_[op.vid]

#         return op

#     def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
#         if op.vid in self.var_remap_:
#             return self.var_remap_[op.vid]

#         return op

#     def visit_function_(self, op: Function) -> Expr:
#         params = []
#         all_params_unchanged = True
#         for param in op.params:
#             new_param = self.visit_var_def(param)
#             params.append(new_param)
#             all_params_unchanged &= param.same_as(new_param)

#         ret_type = self.visit_type(op.ret_type)
#         body = self.visit_with_new_scope(op.body)

#         # TODO(@lesheng): op.ret_type.same_as(ret_type) after Type.same_as is fixed
#         if all_params_unchanged and (op.ret_type == ret_type) and op.body.same_as(body):
#             return op
#         else:
#             return Function(params, body, ret_type, op.attrs, op.span)

#     def visit_if_(self, op: If) -> Expr:
#         guard = self.visit_expr(op.cond)
#         true_b = self.visit_with_new_scope(op.true_branch)
#         false_b = self.visit_with_new_scope(op.false_branch)
#         if (
#             op.cond.same_as(guard)
#             and op.true_branch.same_as(true_b)
#             and op.false_branch.same_as(false_b)
#         ):
#             return op
#         else:
#             return If(guard, true_b, false_b, op.span)

#     def visit_seq_expr_(self, op: SeqExpr) -> Expr:
#         all_blocks_unchanged = True
#         blocks = []
#         for block in op.blocks:
#             new_block = self.visit_binding_block(block)
#             if new_block.bindings:
#                 blocks.append(new_block)
#             all_blocks_unchanged &= block.same_as(new_block)

#         self.builder_._begin_binding_block()
#         body = self.visit_expr(op.body)
#         prologue = self.builder_._end_block()
#         if prologue.bindings:
#             blocks.append(prologue)
#             all_blocks_unchanged = False

#         if all_blocks_unchanged and op.body.same_as(body):
#             return op
#         else:
#             return SeqExpr(blocks, body, op.span)

#     def visit_var_binding_(self, binding: VarBinding) -> None:
#         """Visit VarBinding, a new VarBinding will be emitted

#         Parameters
#         ----------
#         binding: VarBinding
#             The VarBinding to be visited.
#         """
#         new_value = self.visit_expr(binding.value)
#         new_var = self.visit_var_def(binding.var)

#         def emit(b: VarBinding):
#             if self.builder_.current_block_is_dataflow() and not isinstance(b.var, DataflowVar):
#                 self.builder_.emit_output_var_binding(b)
#             else:
#                 self.builder_.emit_var_binding(b)

#         if binding.var.same_as(new_var) and binding.value.same_as(new_value):
#             emit(binding)
#             return

#         temp = self.with_shape_and_type(new_var, new_value.shape_, new_value._checked_type_)
#         if not temp.same_as(new_var):
#             new_var = temp
#             self.var_remap_[binding.var.vid] = new_var

#         emit(VarBinding(new_var, new_value))

#     def visit_match_shape_(self, binding: MatchShape) -> None:
#         """Visit MatchShape, a new MatchShape will be emitted

#         Parameters
#         ----------
#         binding: MatchShape
#             The MatchShape binding to be visited.
#         """
#         new_value = self.visit_expr(binding.value)
#         new_pattern = self.visit_expr(ShapeExpr(binding.pattern))

#         if binding.var:
#             new_shape = None
#             if new_value._checked_type_ and isinstance(new_value._checked_type_, DynTensorType):
#                 new_shape = new_pattern
#             new_var = self.visit_var_def(binding.var)
#             temp = self.with_shape_and_type(new_var, new_shape, new_value._checked_type_)
#             if not temp.same_as(new_var):
#                 new_var = temp
#                 self.var_remap_[binding.var.vid] = new_var

#         if binding.value.same_as(new_value) and binding.pattern.same_as(new_pattern):
#             if not binding.var or (binding.var and binding.var.same_as(new_var)):
#                 self.builder_.match_shape_binding(binding)
#                 return

#         self.builder_.match_shape_binding(MatchShape(new_value, new_pattern.values, new_var))

#     def visit_binding_block_(self, block: BindingBlock) -> BindingBlock:
#         self.builder_._begin_binding_block()
#         for binding in block.bindings:
#             self.visit_binding(binding)
#         return self.builder_._end_block()

#     def visit_dataflow_block_(self, block: DataflowBlock) -> BindingBlock:
#         self.builder_._begin_dataflow_block()
#         for binding in block.bindings:
#             self.visit_binding(binding)
#         return self.builder_._end_block()

#     def visit_dataflow_var_def_(self, var: DataflowVar) -> Var:
#         """Rewrite the dataflow var definition site.

#         Parameters
#         ----------
#         var: DataflowVar
#             The dataflow var to be visited.

#         Returns
#         -------
#         var: Dataflowvar
#             The dataflow var after post-order rewritten.
#         """
#         shape_unchanged = True
#         new_shape = None
#         if var.shape_:
#             new_shape = self.visit_expr(var.shape_)
#             shape_unchanged &= var.shape_.same_as(new_shape)

#         if shape_unchanged:
#             return var
#         else:
#             new_var = DataflowVar(var.vid, None, var._checked_type_, var.span)
#             _update_shape(new_var, new_shape)

#             self.var_remap_[var.vid] = new_var
#             return new_var

#     def visit_var_def_(self, var: Var) -> Var:
#         """Rewrite the var definition site.

#         Parameters
#         ----------
#         var: Var
#             The var to be visited.

#         Returns
#         -------
#         var: Var
#             The var after post-order rewritten.
#         """
#         shape_unchanged = True
#         new_shape = None
#         if var.shape_:
#             new_shape = self.visit_expr(var.shape_)
#             shape_unchanged &= var.shape_.same_as(new_shape)

#         if shape_unchanged:
#             return var
#         else:
#             new_var = Var(var.vid, None, var._checked_type_, var.span)
#             _update_shape(new_var, new_shape)

#             self.var_remap_[var.vid] = new_var
#             return new_var

#     def visit_binding(self, binding: Binding) -> None:
#         if isinstance(binding, MatchShape):
#             self.visit_match_shape_(binding)
#         elif isinstance(binding, VarBinding):
#             self.visit_var_binding_(binding)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(binding)))

#     def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
#         if isinstance(block, DataflowBlock):
#             ret = self.visit_dataflow_block_(block)
#         elif isinstance(block, BindingBlock):
#             ret = self.visit_binding_block_(block)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(block)))

#         return ret

#     def visit_var_def(self, var: Var) -> Var:
#         ret = None
#         if isinstance(var, DataflowVar):
#             ret = self.visit_dataflow_var_def_(var)
#         elif isinstance(var, Var):
#             ret = self.visit_var_def_(var)
#         else:
#             raise TypeError("Invalid type: {0}".format(type(var)))
#         return ret

#     def visit_with_new_scope(self, expr: Expr) -> Expr:
#         self.builder_._begin_binding_block()
#         ret = self.visit_expr(expr)
#         prologue = self.builder_._end_block()
#         if prologue.bindings:
#             ret = SeqExpr([prologue], ret)
#         return ret

#     def with_shape_and_type(self, var: Var, shape: Optional[Expr], t: Type) -> Var:
#         """Create a new var with specified shape and type if the original var's shape or type
#         does not match with the specified ones.

#         Parameters
#         ----------
#         var: Var
#             The var to be updated.
#         shape: Optional[Expr]
#             The specified shape.
#         t: Type
#             The specified type.

#         Returns
#         -------
#         var: Var
#             The var filled with shape and type.
#         """
#         shape_changed = (var.shape_ is not None) ^ (shape is not None)
#         shape_changed |= (
#             var.shape_ and shape and not self.builder_.can_prove_shape_equal(var.shape_, shape)
#         )

#         type_changed = (var._checked_type_ is not None) ^ (t is not None)
#         type_changed |= var._checked_type_ and t and not structural_equal(var._checked_type_, t)

#         if shape_changed or type_changed:
#             new_var = (
#                 DataflowVar(var.vid, None, None, var.span)
#                 if isinstance(var, DataflowVar)
#                 else Var(var.vid, None, None, var.span)
#             )
#             _update_shape(new_var, var.shape_)
#             _update_type(new_var, var._checked_type_)
#             var = new_var

#         if shape_changed:
#             var.shape_ = shape

#         if type_changed:
#             var._checked_type_ = t

#         return var

#     def lookup_binding(self, var: Var) -> Optional[Expr]:
#         return self.builder_.lookup_binding(var)
