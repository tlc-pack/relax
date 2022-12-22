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
# pylint: disable=unused-argument, invalid-name, abstract-method
"""Perform fused multiply-add rewriting in Python"""
from tvm.ir import Op
from tvm.ir.module import IRModule
from tvm.ir.transform import module_pass
from ..expr_functor import mutator, PyExprMutator
from ..expr import Call, Function, Var
from ..transform import dataflowblock_pass


@mutator
class EwiseFMARewriter(PyExprMutator):
    """Rewrites the relax.add call to a relax.ewise_fma call
    when detecting the multiply-add pattern.

    Example
    --------
    x0 = mul(a, b)
    z0 = add(x0, c)
    -->
    z0 = ewise_fma(a, b, c)
    """

    def visit_call_(self, call: Call) -> Call:  # pylint: disable=arguments-differ
        call = self.visit_expr_post_order(call)
        add_op = Op.get("relax.add")
        multiply_op = Op.get("relax.multiply")
        ewise_fma_op = Op.get("relax.ewise_fma")

        if call.op == add_op:
            value = self.lookup_binding(call.args[0])
            if isinstance(value, Call) and value.op == multiply_op:  # type: ignore
                fma_call = Call(
                    ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
                )
                return fma_call

        return call


@dataflowblock_pass(opt_level=2, name="ewise_fma_rewriter")
class EwiseRewriteFMA:
    """The wrapper for the EwiseFMARewriter pass."""

    def transform_dataflowblock(self, block, mod, ctx):
        return EwiseFMARewriter().visit_binding_block(block)


@mutator
class EwiseFuseFMAMutator(PyExprMutator):
    """Performs multiply add fusion. The difference of EwiseFMARewriter and this
    EwiseFuseFMAMutator class is that this mutator generates a sub function(subgraph)
    whose body is a CallNode that calls to the relax.ewise_fma op, and rewrites the
    relax.add call in the main function to calling to the subgraph.

    Example
    --------
    Before-transformation IRModule:
    def main():
        x0 = mul(a, b)
        z0 = add(x0, c)
    -->
    After-transformation IRModule:
    def ewise_fused(x, y, z):
        return relax.ewise_fma(x, y, z)

    def main():
        z0 = ewise_fused(a, b, c)
    """

    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if isinstance(func, Function):
                func = self.visit_expr(func)
            self.builder_.add_func(func, global_var.name_hint)

        return self.builder_.get()

    def visit_call_(self, call: Call) -> Call:  # pylint: disable=arguments-differ
        call = self.visit_expr_post_order(call)
        add_op = Op.get("relax.add")
        multiply_op = Op.get("relax.multiply")
        ewise_fma_op = Op.get("relax.ewise_fma")

        if call.op == add_op:
            value = self.lookup_binding(call.args[0])
            if isinstance(value, Call) and value.op == multiply_op:  # type: ignore
                mul = value
                # construct a subgraph
                x = Var("x", mul.args[0].struct_info)
                y = Var("y", mul.args[1].struct_info)
                z = Var("z", call.args[1].struct_info)
                body = Call(ewise_fma_op, [x, y, z])

                func_name = "ewise_fma_fused"
                func = Function([x, y, z], body, call.args[1].struct_info)
                ewise_fma_fused = func.with_attr("global_symbol", func_name)
                normalized = self.builder_.normalize(ewise_fma_fused)
                global_var = self.builder_.add_func(normalized, "ewise_fma_fused")

                # construct a call to the subgraph
                fma_call = Call(global_var, [mul.args[0], mul.args[1], call.args[1]], None, None)

                return fma_call

        return call


@module_pass(opt_level=2, name="ewise_fuse_fma_rewriter")
class EwiseFuseFMA:
    """The wrapper for the EwiseFuseFMA pass."""

    def transform_module(self, mod, ctx):
        return EwiseFuseFMAMutator(mod).transform()
