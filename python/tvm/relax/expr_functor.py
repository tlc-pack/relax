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
from ..relay import Id
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

    def visit_expr(self, expr: Expr) -> None:
        return _ffi_api.PyExprVisitorVisitExpr(self, expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprVisitorVisitBinding(self, binding)

    def visit_binding_block(self, block: BindingBlock) -> None:
        return _ffi_api.PyExprVisitorVisitBindingBlock(self, block)

    def visit_var_def(self, var: Var) -> None:
        return _ffi_api.PyExprVisitorVisitVarDef(self, var)


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

    builder_: BlockBuilder

    def __init__(self, packed_value) -> None:
        self.__init_handle_by_constructor__(_ffi_api.MakeExprMutator, *packed_value)

    def set_var_remap(self, id: Id, var: Var) -> None:
        return _ffi_api.PyExprMutatorSetVarRemap(self, id, var)

    def get_var_remap(self, id: Id) -> Var:
        return _ffi_api.PyExprMutatorGetVarRemap(self, id)

    def visit_expr(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitExpr(self, expr)

    def visit_binding(self, binding: Binding) -> None:
        return _ffi_api.PyExprMutatorVisitBinding(self, binding)

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        return _ffi_api.PyExprMutatorVisitBindingBlock(self, block)

    def visit_var_def(self, var: Var) -> Var:
        return _ffi_api.PyExprMutatorVisitVarDef(self, var)

    def visit_with_new_scope(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitWithNewScope(self, expr)

    def lookup_binding(self, var: Var) -> Optional[Expr]:
        return _ffi_api.PyExprMutatorLookupBinding(self, var)

    def with_shape_and_type(self, var: Var, shape: None, t: Type) -> Var:  # TODO: shape anno
        return _ffi_api.PyExprMutatorWithShapeAndType(self, var, shape, t)


class PyExprMutator:
    def __init__(self) -> None:
        """"""

    def __getattr__(self, name):
        if name in ["builder_"]:
            return getattr(self._outer(), name)
        else:
            return self.__getattribute__(name)

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

    def visit_with_new_scope(self, expr: Expr) -> Expr:
        return _ffi_api.PyExprMutatorVisitWithNewScope(self._outer(), expr)

    def lookup_binding(self, var: Var) -> Optional[Expr]:
        return _ffi_api.PyExprMutatorLookupBinding(self._outer(), var)

    def with_shape_and_type(self, var: Var, shape: None, t: Type) -> Var:  # TODO: shape anno
        return _ffi_api.PyExprMutatorWithShapeAndType(self._outer(), var, shape, t)


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

            super().__init__(packed_value)

            self._inst._outer = weakref.ref(self)
            self.builder_ = super().__getattr__("builder_")

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
