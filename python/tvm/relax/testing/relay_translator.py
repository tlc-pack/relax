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
"""Relay to Relax translator."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import tvm
from tvm.ir.module import IRModule
from tvm import relax, relay
from tvm.relax.testing import nn
from tvm.relay.backend.te_compiler import select_implementation
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.meta_schedule.utils import autotvm_silencer


def from_relay(
    func: relay.Function,
    target: Target,
    relay_params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Optional[Dict[str, Any]] = None,
    disabled_pass: Optional[List[str]] = None,
) -> IRModule:
    """Convert a Relay function into a Relax program.

    Parameters
    ----------
    func : relay.Function
        Relay function to be converted

    Returns
    -------
    mod : tvm.IRModule
        The Relax IRModule for compilation
    """
    # A map to store the mapping of Relay Expr to its corresponding Relax var
    var_map = {}
    # The output of the function
    output_var = None

    if not isinstance(target, Target):
        target = Target(target)
    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
            "relay.backend.use_meta_schedule": True,
        }

    if relay_params:
        func = relay.build_module.bind_params_by_name(func, relay_params)

    params = []

    def convert_shape(shape: List[tvm.tir.PrimExpr]) -> List[tvm.tir.PrimExpr]:
        """Convert the relay shape to relax shape by changing Any dim to symbolic dim"""
        ret = []
        for dim in shape:
            if isinstance(dim, tvm.tir.IntImm):
                ret.append(tvm.tir.IntImm("int64", int(dim)))
            elif isinstance(dim, tvm.tir.Any):
                ret.append(tvm.tir.Var("d", "int64"))
            else:
                ret.append(dim)
        return ret

    def visit_func(node):
        nonlocal output_var
        if isinstance(node, relay.Var):
            if isinstance(node.type_annotation, relay.TensorType):
                var_map[node] = nn.Placeholder(
                    tuple(convert_shape(node.type_annotation.shape)),
                    node.type_annotation.dtype,
                    node.name_hint,
                )
                params.append(var_map[node])
            else:
                raise TypeError("The type of relay.Var to be translated must be of TensorType.")
        elif isinstance(node, relay.Call):
            args = node.args
            new_args = []
            te_inputs = []
            for arg in args:
                if arg in var_map:
                    new_args.append(var_map[arg])
                    te_inputs.append(tvm.relax.expr.te_tensor(new_args[-1]))

            op_name = node.op.name
            attrs = node.attrs
            out_type = node.checked_type

            best_impl, outputs = select_implementation(
                node.op,
                attrs,
                te_inputs,
                out_type,
                target,
                use_autotvm=False,
            )
            compute_func = best_impl.compute
            name_hint = op_name.split(".")[-1]
            var = bb.emit_te(
                compute_func, attrs, new_args, node.checked_type, primfunc_name_hint=name_hint
            )
            output_var = var
            var_map[node] = var
        elif isinstance(node, relay.Constant):
            # fill the shape and checked_type fields of the Constant
            new_constant = relay.Constant(node.data)
            var_map[node] = new_constant
        elif isinstance(node, relay.Tuple):
            new_fields = []
            for field in node.fields:
                if field in var_map:
                    new_fields.append(var_map[field])
                else:
                    raise RuntimeError("field is not in var_map.")
            new_tuple = relax.Tuple(new_fields)
            new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
            var_map[node] = new_tuple_var
            output_var = new_tuple_var
        elif isinstance(node, relay.TupleGetItem):
            if node.tuple_value in var_map:
                new_tuple = var_map[node.tuple_value]
                new_tuple_get_item_node = relax.TupleGetItem(new_tuple, node.index)
                new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
                var_map[node] = new_tuple_get_item_var
                output_var = new_tuple_get_item_var
            else:
                raise RuntimeError("tuple is not in var_map")
        elif isinstance(node, relay.Function):
            relax.BlockBuilder.current().emit_func_output(output_var, params)
        elif isinstance(node, tvm.ir.Op):
            pass
        else:
            raise TypeError("{} is not supported yet.".format(str(type(node))))

    # List of subset of relay->relay optimizations
    # See src/relay/backend/utils.cc::GetPassPrefix() for full list
    seq = tvm.get_global_func("relay.backend.GetPassPrefixSeq")(True, True)

    # Since optimization passes and OpStrategy are highly context-dependent,
    # we match the exact same context with `extract_task_from_relay()` env
    with autotvm_silencer(), target, tvm.transform.PassContext(
        opt_level=opt_level, config=pass_config, disabled_pass=disabled_pass
    ):
        mod = tvm.IRModule.from_expr(func)
        mod = seq(mod)
        bb = relax.BlockBuilder()
        with bb.function("main"):
            relay.analysis.post_order_visit(mod["main"], visit_func)

    return bb.get()
