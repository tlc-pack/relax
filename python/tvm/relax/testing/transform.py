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
# pylint: disable=unused-argument, invalid-name, no-else-return
"""Relax transformation passes for testing"""

from __future__ import annotations
from tvm import ir
from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.target import Target
from tvm.ir import transform
from tvm.relax import ExprMutator
from tvm.relax.expr import Call
from tvm.relay.backend.te_compiler import select_implementation


@ir.transform.module_pass(opt_level=0)
class LowerWithRelayOpStrategyPass(transform.Pass):
    """Lower Relax Op into TIR by using Relay OpStrategy.

    Since operators like conv2d, add, matmul are relay-, relax- independent,
    this pass assumes we can always find relay op equivalent for such relax ops,
    and use Relay Op Strategy (legacy) to perform lowering and find the TOPI implementation.

    Parameters
    ----------
    target : Target
        target info

    Returns
    -------
    pass : transform.Pass
        lowering pass
    """

    def __init__(self, target: Target):
        self.target = target

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        """Implement lowering mechanism.

        Parameters
        ----------
        mod : IRModule
            Input IRModule with Relax ops

        ctx: PassContext
            Pass context

        Returns
        -------
        out_mod : IRModule
            Output IRModule with lowered TIR functions
        """
        target = self.target

        class Lowerer(ExprMutator):
            """Mutator that performs lowering."""

            def visit_call_(self, call_node: Call):
                # Remove "relax." prefix to deduce relay op name
                relay_op_name = call_node.op.name[6:]
                # Check if equivalent relay op exists. If not, return the original call.
                if relay_op_name in ir.Op.list_op_names():
                    relay_op = ir.Op.get(relay_op_name)

                    te_inputs = [relax.expr.te_tensor(arg) for arg in call_node.args]
                    best_impl_tuple = select_implementation(
                        relay_op,
                        call_node.attrs,
                        te_inputs,
                        call_node.checked_type,
                        target,
                        use_autotvm=False,
                    )
                    compute_func = best_impl_tuple[0].compute
                    name_hint = relay_op_name.split(".")[-1]

                    return self.builder_.call_te(
                        compute_func,
                        call_node.attrs,
                        call_node.args,
                        call_node.attrs,
                        primfunc_name_hint=name_hint,
                    )
                else:
                    return call_node

            # TOOD(@team): transform() wapper is necessary to include TIR functions.
            # IMO, this is bit unintuitive. Can we improve this?
            def transform(self):
                for gv, func in mod.functions.items():
                    if isinstance(func, relax.Function):
                        updated_func = self.visit_expr(func)
                        self.builder_.update_func(gv, updated_func)
                return self.builder_.get()

        return Lowerer().transform()
