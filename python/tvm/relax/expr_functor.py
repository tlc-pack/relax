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
from typing import Optional, Callable
from tvm.ir import Op
from tvm.meta_schedule.utils import derived_object

from .expr import Type, Span, Expr
from .expr import Function, ExternFunc
from .expr import Constant, Var, DataflowVar
from .expr import ShapeExpr, RuntimeDepShape
from .expr import GlobalVar, SeqExpr, Tuple
from .expr import Call, If, TupleGetItem
from .expr import Binding, MatchShape, VarBinding
from .expr import BindingBlock, DataflowBlock
from ..relay import Id
from .block_builder import BlockBuilder
from . import _ffi_api

visitor = derived_object
mutator = derived_object


@tvm._ffi.register_object("expr_functor.PyExprVisitor")
class _PyExprVisitor(tvm.runtime.Object):
    """TODO"""

    def __init__(
        self,
        f_visit_expr: Callable = None,
        f_visit_constant_: Callable = None,
        f_visit_tuple_: Callable = None,
        f_visit_var_: Callable = None,
        f_visit_dataflow_var_: Callable = None,
        f_visit_shape_expr_: Callable = None,
        f_visit_runtime_dep_shape_: Callable = None,
        f_visit_extern_func_: Callable = None,
        f_visit_global_var_: Callable = None,
        f_visit_function_: Callable = None,
        f_visit_call_: Callable = None,
        f_visit_seq_expr_: Callable = None,
        f_visit_if_: Callable = None,
        f_visit_op_: Callable = None,
        f_visit_tuple_getitem_: Callable = None,
        f_visit_binding: Callable = None,
        f_visit_var_binding_: Callable = None,
        f_visit_match_shape_: Callable = None,
        f_visit_binding_block: Callable = None,
        f_visit_binding_block_: Callable = None,
        f_visit_dataflow_block_: Callable = None,
        f_visit_var_def: Callable = None,
        f_visit_var_def_: Callable = None,
        f_visit_dataflow_var_def_: Callable = None,
        f_visit_type: Callable = None,
        f_visit_span: Callable = None,
    ) -> None:
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MakePyExprVisitor,
            f_visit_expr,
            f_visit_constant_,
            f_visit_tuple_,
            f_visit_var_,
            f_visit_dataflow_var_,
            f_visit_shape_expr_,
            f_visit_runtime_dep_shape_,
            f_visit_extern_func_,
            f_visit_global_var_,
            f_visit_function_,
            f_visit_call_,
            f_visit_seq_expr_,
            f_visit_if_,
            f_visit_op_,
            f_visit_tuple_getitem_,
            f_visit_binding,
            f_visit_var_binding_,
            f_visit_match_shape_,
            f_visit_binding_block,
            f_visit_binding_block_,
            f_visit_dataflow_block_,
            f_visit_var_def,
            f_visit_var_def_,
            f_visit_dataflow_var_def_,
            f_visit_type,
            f_visit_span,
        )

    def visit_expr(self, expr: Expr) -> None:
        return _ffi_api.PyExprVisitorVisitExpr(self, expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprVisitorVisitBinding(self, binding)

    def visit_binding_block(self, block: BindingBlock) -> None:
        return _ffi_api.PyExprVisitorVisitBindingBlock(self, block)

    def visit_var_def(self, var: Var) -> None:
        return _ffi_api.PyExprVisitorVisitVarDef(self, var)


class PyExprVisitor:
    _tvm_metadata = {
        "cls": _PyExprVisitor,
        "methods": [
            "visit_expr",
            "visit_constant_",
            "visit_tuple_",
            "visit_var_",
            "visit_dataflow_var_",
            "visit_shape_expr_",
            "visit_runtime_dep_shape_",
            "visit_extern_func_",
            "visit_global_var_",
            "visit_function_",
            "visit_call_",
            "visit_seq_expr_",
            "visit_if_",
            "visit_op_",
            "visit_tuple_getitem_",
            "visit_binding",
            "visit_var_binding_",
            "visit_match_shape_",
            "visit_binding_block",
            "visit_binding_block_",
            "visit_dataflow_block_",
            "visit_var_def",
            "visit_var_def_",
            "visit_dataflow_var_def_",
            "visit_type",
            "visit_span",
        ],
    }

    def visit_expr(self, expr: Expr) -> None:
        return _ffi_api.PyExprVisitorVisitExpr(self._outer(), expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprVisitorVisitBinding(self._outer(), binding)

    def visit_binding_block(self, block: BindingBlock) -> None:
        return _ffi_api.PyExprVisitorVisitBindingBlock(self._outer(), block)

    def visit_var_def(self, var: Var) -> None:
        return _ffi_api.PyExprVisitorVisitVarDef(self._outer(), var)

    def visit_constant_(self, op: Constant) -> None:
        raise NotImplementedError

    def visit_tuple_(self, op: Tuple) -> None:
        raise NotImplementedError

    def visit_var_(self, op: Var) -> None:
        raise NotImplementedError

    def visit_dataflow_var_(self, op: DataflowVar) -> None:
        raise NotImplementedError

    def visit_shape_expr_(self, op: ShapeExpr) -> None:
        raise NotImplementedError

    def visit_runtime_dep_shape_(self, op: RuntimeDepShape) -> None:
        raise NotImplementedError

    def visit_extern_func_(self, op: ExternFunc) -> None:
        raise NotImplementedError

    def visit_global_var_(self, op: GlobalVar) -> None:
        raise NotImplementedError

    def visit_function_(self, op: Function) -> None:
        raise NotImplementedError

    def visit_call_(self, op: Call) -> None:
        raise NotImplementedError

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        raise NotImplementedError

    def visit_if_(self, op: If) -> None:
        raise NotImplementedError

    def visit_op_(self, op: Op) -> None:
        raise NotImplementedError

    def visit_tuple_getitem_(self, op: TupleGetItem) -> None:
        raise NotImplementedError

    def visit_var_binding_(self, binding: VarBinding) -> None:
        raise NotImplementedError

    def visit_match_shape_(self, binding: MatchShape) -> None:
        raise NotImplementedError

    def visit_binding_block_(self, block: BindingBlock) -> None:
        raise NotImplementedError

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        raise NotImplementedError

    def visit_var_def_(self, var: Var) -> None:
        raise NotImplementedError

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        raise NotImplementedError

    def visit_type(self, t: Type) -> None:
        raise NotImplementedError

    def visit_span(self, span: Span) -> None:
        raise NotImplementedError


@tvm._ffi.register_object("expr_functor.PyExprMutator")
class _PyExprMutator(tvm.runtime.Object):
    """TODO"""

    builder_: BlockBuilder

    def __init__(
        self,
        builder: BlockBuilder = None,
        f_visit_expr: Callable = None,
        f_visit_constant_: Callable = None,
        f_visit_tuple_: Callable = None,
        f_visit_var_: Callable = None,
        f_visit_dataflow_var_: Callable = None,
        f_visit_shape_expr_: Callable = None,
        f_visit_runtime_dep_shape_: Callable = None,
        f_visit_extern_func_: Callable = None,
        f_visit_global_var_: Callable = None,
        f_visit_function_: Callable = None,
        f_visit_call_: Callable = None,
        f_visit_seq_expr_: Callable = None,
        f_visit_if_: Callable = None,
        f_visit_op_: Callable = None,
        f_visit_tuple_getitem_: Callable = None,
        f_visit_binding: Callable = None,
        f_visit_var_binding_: Callable = None,
        f_visit_match_shape_: Callable = None,
        f_visit_binding_block: Callable = None,
        f_visit_binding_block_: Callable = None,
        f_visit_dataflow_block_: Callable = None,
        f_visit_var_def: Callable = None,
        f_visit_var_def_: Callable = None,
        f_visit_dataflow_var_def_: Callable = None,
        f_visit_type: Callable = None,
        f_visit_span: Callable = None,
        f_rewrite_constant_post_order: Callable = None,
        f_rewrite_tuple_post_order: Callable = None,
        f_rewrite_var_post_order: Callable = None,
        f_rewrite_dataflow_var_post_order: Callable = None,
        f_rewrite_shape_expr_post_order: Callable = None,
        f_rewrite_runtime_dep_shape_post_order: Callable = None,
        f_rewrite_extern_func_post_order: Callable = None,
        f_rewrite_global_var_post_order: Callable = None,
        f_rewrite_function_post_order: Callable = None,
        f_rewrite_call_post_order: Callable = None,
        f_rewrite_seq_expr_post_order: Callable = None,
        f_rewrite_if_post_order: Callable = None,
        f_rewrite_op_post_order: Callable = None,
        f_rewrite_tuple_getitem_post_order: Callable = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MakePyExprMutator,
            builder,
            f_visit_expr,
            f_visit_constant_,
            f_visit_tuple_,
            f_visit_var_,
            f_visit_dataflow_var_,
            f_visit_shape_expr_,
            f_visit_runtime_dep_shape_,
            f_visit_extern_func_,
            f_visit_global_var_,
            f_visit_function_,
            f_visit_call_,
            f_visit_seq_expr_,
            f_visit_if_,
            f_visit_op_,
            f_visit_tuple_getitem_,
            f_visit_binding,
            f_visit_var_binding_,
            f_visit_match_shape_,
            f_visit_binding_block,
            f_visit_binding_block_,
            f_visit_dataflow_block_,
            f_visit_var_def,
            f_visit_var_def_,
            f_visit_dataflow_var_def_,
            f_visit_type,
            f_visit_span,
            f_rewrite_constant_post_order,
            f_rewrite_tuple_post_order,
            f_rewrite_var_post_order,
            f_rewrite_dataflow_var_post_order,
            f_rewrite_shape_expr_post_order,
            f_rewrite_runtime_dep_shape_post_order,
            f_rewrite_extern_func_post_order,
            f_rewrite_global_var_post_order,
            f_rewrite_function_post_order,
            f_rewrite_call_post_order,
            f_rewrite_seq_expr_post_order,
            f_rewrite_if_post_order,
            f_rewrite_op_post_order,
            f_rewrite_tuple_getitem_post_order,
        )

    def visit_expr(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitExpr(self, expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprMutatorVisitBinding(self, binding)

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        return _ffi_api.PyExprMutatorVisitBindingBlock(self, block)

    def visit_var_def(self, var: Var) -> Var:
        return _ffi_api.PyExprMutatorVisitVarDef(self, var)


class PyExprMutator:
    _tvm_metadata = {
        "cls": _PyExprMutator,
        "fields": ["builder_"],
        "methods": [
            "visit_expr",
            "visit_constant_",
            "visit_tuple_",
            "visit_var_",
            "visit_dataflow_var_",
            "visit_shape_expr_",
            "visit_runtime_dep_shape_",
            "visit_extern_func_",
            "visit_global_var_",
            "visit_function_",
            "visit_call_",
            "visit_seq_expr_",
            "visit_if_",
            "visit_op_",
            "visit_tuple_getitem_",
            "visit_binding",
            "visit_var_binding_",
            "visit_match_shape_",
            "visit_binding_block",
            "visit_binding_block_",
            "visit_dataflow_block_",
            "visit_var_def",
            "visit_var_def_",
            "visit_dataflow_var_def_",
            "visit_type",
            "visit_span",
            "rewrite_constant_post_order",
            "rewrite_tuple_post_order",
            "rewrite_var_post_order",
            "rewrite_dataflow_var_post_order",
            "rewrite_shape_expr_post_order",
            "rewrite_runtime_dep_shape_post_order",
            "rewrite_extern_func_post_order",
            "rewrite_global_var_post_order",
            "rewrite_function_post_order",
            "rewrite_call_post_order",
            "rewrite_seq_expr_post_order",
            "rewrite_if_post_order",
            "rewrite_op_post_order",
            "rewrite_tuple_getitem_post_order",
        ],
    }

    def __init__(self) -> None:
        self.builder_ = BlockBuilder()

    def set_var_remap(self, id: Id, var: Var) -> None:
        return _ffi_api.PyExprMutatorSetVarRemap(self._outer(), id, var)

    def get_var_remap(self, id: Id) -> Var:
        return _ffi_api.PyExprMutatorGetVarRemap(self._outer(), id)

    def visit_expr(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitExpr(self._outer(), expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprMutatorVisitBinding(self._outer(), binding)

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        return _ffi_api.PyExprMutatorVisitBindingBlock(self._outer(), block)

    def visit_var_def(self, var: Var) -> Var:
        return _ffi_api.PyExprMutatorVisitVarDef(self._outer(), var)

    def visit_constant_(self, op: Constant) -> Expr:
        raise NotImplementedError

    def visit_tuple_(self, op: Tuple) -> Expr:
        raise NotImplementedError

    def visit_var_(self, op: Var) -> Expr:
        raise NotImplementedError

    def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
        raise NotImplementedError

    def visit_shape_expr_(self, op: ShapeExpr) -> Expr:
        raise NotImplementedError

    def visit_runtime_dep_shape_(self, op: RuntimeDepShape) -> Expr:
        raise NotImplementedError

    def visit_extern_func_(self, op: ExternFunc) -> Expr:
        raise NotImplementedError

    def visit_global_var_(self, op: GlobalVar) -> Expr:
        raise NotImplementedError

    def visit_function_(self, op: Function) -> Expr:
        raise NotImplementedError

    def visit_call_(self, op: Call) -> Expr:
        raise NotImplementedError

    def visit_seq_expr_(self, op: SeqExpr) -> Expr:
        raise NotImplementedError

    def visit_if_(self, op: If) -> Expr:
        raise NotImplementedError

    def visit_op_(self, op: Op) -> Expr:
        raise NotImplementedError

    def visit_tuple_getitem_(self, op: TupleGetItem) -> Expr:
        raise NotImplementedError

    def visit_var_binding_(self, binding: VarBinding) -> None:
        raise NotImplementedError

    def visit_match_shape_(self, binding: MatchShape) -> None:
        raise NotImplementedError

    def visit_binding_block_(self, block: BindingBlock) -> BindingBlock:
        raise NotImplementedError

    def visit_dataflow_block_(self, block: DataflowBlock) -> BindingBlock:
        raise NotImplementedError

    def visit_var_def_(self, var: Var) -> Var:
        raise NotImplementedError

    def visit_dataflow_var_def_(self, var: DataflowVar) -> Var:
        raise NotImplementedError

    def visit_type(self, t: Type) -> Type:
        raise NotImplementedError

    def visit_span(self, span: Span) -> Span:
        raise NotImplementedError

    def rewrite_constant_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_tuple_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_var_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_dataflow_var_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_shape_expr_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_runtime_dep_shape_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_extern_func_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_global_var_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_function_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_call_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_seq_expr_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_if_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_op_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def rewrite_tuple_getitem_post_order(self, op: Constant) -> Expr:
        raise NotImplementedError

    def visit_with_new_scope(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitWithNewScope(self._outer(), expr)

    def lookup_binding(self, var: Var) -> Optional[Expr]:
        return _ffi_api.PyExprMutatorLookupBinding(self._outer(), var)

    def with_shape_and_type(self, var: Var, shape: None, t: Type) -> Var:  # TODO: shape anno
        return _ffi_api.PyExprMutatorWithShapeAndType(self._outer(), var, shape, t)
