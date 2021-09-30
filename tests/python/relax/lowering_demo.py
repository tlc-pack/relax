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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import relax as rx
from tvm import tir, relay
from tvm.relay import Call
from tvm.ir import structural_equal
import numpy as np
from termcolor import colored


# Before call_dps lowering

# @tvm.script.relax
# class Module:
#     def foo(x: Tensor[(3, 4), "float32"]):
#         with relax.dataflow():
#             gv0 = relax.call_dps((3, 4), "test.op.identity", (x,))
#             relax.output(gv0)
#         return gv0

def original_program():
    shape_anno = [3, 4]
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", shape_anno, type_anno)
    ib = rx.IRBuilder()
    with ib.function(x, "foo"):
        with ib.dataflow() as df:
            lv0 = rx.call_dps([3, 4], rx.ExternFunc("test.op.identity"), [x])
            gv0 = ib.emit_output(lv0)
        ib.emit_output(gv0)
    expr = ib.get()
    return expr

# after call_dps lowering

# @rx.script
# def foo(x: Tensor[(3, 4), "float32"]):
#   gv0 = relax.call_packed("relax.builtin.alloc_tensor", (3, 4))
#   relax.call_packed("test.op.identity", (x, gv0))
#   return gv0

def explicit_memory_rewrite():
    print(colored("Original Relax program:", "green"))
    func = original_program()
    mod = tvm.IRModule.from_expr(func)
    print(rx.parser.astext(mod))
    mem_lowered_func = rx.transform.explicit_memory_rewrite(func)
    new_mod = tvm.IRModule.from_expr(mem_lowered_func)
    # print(new_mod.astext())
    # print(rx.parser.astext(new_mod))

# After furthur lowering

# @tvm.script.relax
# class Module:
#     def foo(x: Tensor[(3, 4), "float32"]):
#         gv0 = relax.call_packed("vm.builtin.alloc_storage", (12,), (8,), device_id=0, device_type=1)
#         gv1 = relax.call_packed("vm.builtin.alloc_tensor", gv0, (0,), (3, 4))
#         gv2 = relax.call_packed("test.op.identity", x, gv1)
#         return gv1


@tvm.register_func("test.op.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.asnumpy())

def relax_compile_vm():
    shape_anno = [3, 4]
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", shape_anno, type_anno)
    ib = rx.IRBuilder()
    storage_attr = tvm.ir.attrs.make_node(
        "relax.attrs.AllocStorageAttrs", device_id=0, device_type=1
    )
    tensor_attr = tvm.ir.attrs.make_node("relax.attrs.AllocTensorAttrs")

    # Construct the lowest form program
    with ib.function(x, "foo"):
        gv0 = ib.emit(Call(rx.ExternFunc("vm.builtin.alloc_storage"),[rx.ShapeExpr([12]), rx.ShapeExpr([8])], storage_attr))
        gv1 = ib.emit(Call(rx.ExternFunc("vm.builtin.alloc_tensor"),[gv0, rx.ShapeExpr([0]), rx.ShapeExpr([3, 4])], tensor_attr))
        ib.emit(Call(rx.ExternFunc("test.op.identity"), [x, gv1]))
        ib.emit_output(gv1)
    expr = ib.get()

    mod = tvm.IRModule.from_expr(expr)
    print(colored("After call_dps lowering:", "green"))
    print(rx.parser.astext(mod))

    print(colored("Compile into a VM executable:", "green"))
    exec = rx.transform.compile(expr)
    print(exec.astext())

    print(colored("Run the executable on the VM:", "green"))
    input = tvm.nd.array(np.random.rand(3,4).astype(np.float32))
    print("input array:", input)
    vm = rx.VirtualMachine(exec, tvm.cpu())
    res = vm["foo"](input)
    print("output array:", res)


if __name__ == "__main__":
    explicit_memory_rewrite()
    print("""
@tvm.register_func("test.op.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.asnumpy())
        """)
    relax_compile_vm()
