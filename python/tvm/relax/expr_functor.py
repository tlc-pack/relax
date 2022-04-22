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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression functor of Relax."""
from tvm.ir import Op
from .expr import Function, ExternFunc
from .expr import Constant, Var, DataflowVar
from .expr import ShapeExpr, RuntimeDepShape
from .expr import GlobalVar, SeqExpr, Tuple
from .expr import Call, If, TupleGetItem
from .expr import MatchShape, VarBinding
from .expr import BindingBlock, DataflowBlock


class ExprFunctor:
    """
    An abstract visitor defined over Expr.

    Defines the default dispatch over expressions, and
    implements memoization.
    """

    # pylint: disable=no-else-return
    def visit(self, expr):
        """Apply the visitor to an expression."""
        if isinstance(expr, Function):
            res = self.visit_function(expr)
        elif isinstance(expr, ExternFunc):
            res = self.visit_extern_func(expr)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr)
        elif isinstance(expr, DataflowVar):
            res = self.visit_dataflow_var(expr)
        elif isinstance(expr, Var):
            res = self.visit_var(expr)
        elif isinstance(expr, ShapeExpr):
            res = self.visit_shape_expr(expr)
        elif isinstance(expr, RuntimeDepShape):
            res = self.visit_runtime_dep_shape(expr)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr)
        elif isinstance(expr, SeqExpr):
            res = self.visit_seq_expr(expr)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr)
        elif isinstance(expr, Call):
            res = self.visit_call(expr)
        elif isinstance(expr, If):
            res = self.visit_if(expr)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr)
        elif isinstance(expr, MatchShape):
            res = self.visit_match_shape(expr)
        elif isinstance(expr, VarBinding):
            res = self.visit_var_binding(expr)
        elif isinstance(expr, DataflowBlock):
            res = self.visit_dataflow_block(expr)
        elif isinstance(expr, BindingBlock):
            res = self.visit_binding_block(expr)
        elif isinstance(expr, Op):
            res = self.visit_op(expr)
        else:
            raise Exception("warning unhandled case: {0}".format(type(expr)))

        return res

    def visit_function(self, _):
        raise NotImplementedError()

    def visit_extern_func(self, _):
        raise NotImplementedError()

    def visit_constant(self, _):
        raise NotImplementedError()

    def visit_var(self, _):
        raise NotImplementedError()

    def visit_dataflow_var(self, _):
        raise NotImplementedError()

    def visit_shape_expr(self, _):
        raise NotImplementedError()

    def visit_runtime_dep_shape(self, _):
        raise NotImplementedError()

    def visit_global_var(self, _):
        raise NotImplementedError()

    def visit_seq_expr(self, _):
        raise NotImplementedError()

    def visit_tuple(self, _):
        raise NotImplementedError()

    def visit_call(self, _):
        raise NotImplementedError()

    def visit_if(self, _):
        raise NotImplementedError()

    def visit_tuple_getitem(self, _):
        raise NotImplementedError()

    def visit_match_shape(self, _):
        raise NotImplementedError()

    def visit_var_binding(self, _):
        raise NotImplementedError()

    def visit_dataflow_block(self, _):
        raise NotImplementedError()

    def visit_binding_block(self, _):
        raise NotImplementedError()

    def visit_op(self, _):
        raise NotImplementedError()


class ExprVisitor(ExprFunctor):
    """
    A visitor over Expr.

    The default behavior recursively traverses the AST.
    """

    def visit_function(self, func: Function) -> None:
        for param in func.params:
            self.visit(param)
        self.visit(func.body)

    def visit_extern_func(self, op: ExternFunc) -> None:
        pass

    def visit_constant(self, op: Constant) -> None:
        pass

    def visit_var(self, op: Var) -> None:
        pass

    def visit_dataflow_var(self, op: DataflowVar) -> None:
        pass

    def visit_shape_expr(self, op: ShapeExpr) -> None:
        pass

    def visit_runtime_dep_shape(self, op: RuntimeDepShape) -> None:
        pass

    def visit_global_var(self, op: GlobalVar) -> None:
        pass

    def visit_seq_expr(self, op: SeqExpr) -> None:
        for block in op.blocks:
            self.visit(block)
        self.visit(op.body)

    def visit_tuple(self, op: Tuple) -> None:
        for field in op.fields:
            self.visit(field)

    def visit_call(self, op: Call) -> None:
        self.visit(op.op)
        for arg in op.args:
            self.visit(arg)

    def visit_if(self, op: If) -> None:
        self.visit(op.cond)
        self.visit(op.true_branch)
        self.visit(op.false_branch)

    def visit_tuple_getitem(self, op: TupleGetItem) -> None:
        self.visit(op.tuple_value)

    def visit_match_shape(self, binding: MatchShape) -> None:
        self.visit(binding.value)
        self.visit(binding.var)

    def visit_var_binding(self, binding: VarBinding) -> None:
        self.visit(binding.value)
        self.visit(binding.var)

    def visit_dataflow_block(self, block: DataflowBlock) -> None:
        for binding in block.bindings:
            self.visit(binding)

    def visit_binding_block(self, block: BindingBlock) -> None:
        for binding in block.bindings:
            self.visit(binding)

    def visit_op(self, op: Op) -> None:
        pass


class ExprMutator(ExprFunctor):
    """
    A functional mutator over Expr.

    The default behavior recursively traverses the AST
    and reconstructs the AST.
    """

    def visit_function(self, func: Function) -> Function:
        new_params = [self.visit(param) for param in func.params]
        new_body = self.visit(func.body)
        return Function(new_params, new_body, func.ret_type, func.name, func.span)

    def visit_extern_func(self, op: ExternFunc) -> ExternFunc:
        return op

    def visit_constant(self, op: Constant) -> Constant:
        return op

    def visit_var(self, op: Var) -> Var:
        return op

    def visit_dataflow_var(self, op: DataflowVar) -> DataflowVar:
        return op

    def visit_shape_expr(self, op: ShapeExpr) -> ShapeExpr:
        return op

    def visit_runtime_dep_shape(self, op: RuntimeDepShape) -> RuntimeDepShape:
        return op

    def visit_global_var(self, op: GlobalVar) -> GlobalVar:
        return op

    def visit_seq_expr(self, op: SeqExpr) -> SeqExpr:
        new_blocks = [self.visit(block) for block in op.blocks]
        new_body = self.visit(op.body)
        return SeqExpr(new_blocks, new_body, op.span)

    def visit_tuple(self, op: Tuple) -> Tuple:
        new_fields = [self.visit(field) for field in op.fields]
        return Tuple(new_fields, op.span)

    def visit_call(self, op: Call) -> Call:
        new_op = self.visit(op.op)
        new_args = [self.visit(arg) for arg in op.args]
        return Call(new_op, new_args, op.attrs, op.type_args, op.span)

    def visit_if(self, op: If) -> If:
        new_cond = op.cond
        new_true_branch = op.true_branch
        new_false_branch = op.false_branch
        return If(new_cond, new_true_branch, new_false_branch, op.span)

    def visit_tuple_getitem(self, op: TupleGetItem) -> TupleGetItem:
        new_tuple_value = self.visit(op.tuple_value)
        return TupleGetItem(new_tuple_value, op.index)

    def visit_match_shape(self, binding: MatchShape) -> MatchShape:
        new_value = self.visit(binding.value)
        new_var = self.visit(binding.var)
        return MatchShape(new_value, binding.pattern, new_var, binding.span)

    def visit_var_binding(self, binding: VarBinding) -> VarBinding:
        new_value = self.visit(binding.value)
        new_var = self.visit(binding.var)
        return VarBinding(new_var, new_value, binding.span)

    def visit_dataflow_block(self, block: DataflowBlock) -> DataflowBlock:
        new_bindings = [self.visit(binding) for binding in block.bindings]
        return DataflowBlock(new_bindings, block.span)

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        new_bindings = [self.visit(binding) for binding in block.bindings]
        return BindingBlock(new_bindings, block.span)

    def visit_op(self, op: Op) -> Op:
        return op
