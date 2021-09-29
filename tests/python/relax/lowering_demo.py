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


# def rx_func(func):
#     return func.module[func.fn_name]

# Before memory lowering

# @rx.script
# def foo(x: Tensor[(3, 4), "float32"]):
#   with relax.dataflow():
#       z: Tensor[(3, 4), "float32"] = relax.call_dps((3, 4), rx.extern("test.op.identity"), (x))
#       relax.output(x)
#   return z
# f = rx_func(foo)


# def original_program():
#     shape_anno = [3, 4]
#     type_anno = rx.DynTensorType(2, "float32")
#     x = rx.Var("x", shape_anno, type_anno)
#     ib = rx.IRBuilder()
#     with ib.function(x):
#         with ib.dataflow() as df:
#             lv0 = rx.call_dps([3, 4], rx.extern("test.op.identity"), [x])
#             gv0 = ib.emit_output(lv0)
#         ib.emit_output(gv0)
#     expr = ib.get()


# after rewrite
# func = rx.transform.explicit_memory_rewrite(expr)

# After memory lowering

# @rx.script
# def foo(x: Tensor[(3, 4), "float32"]):
#
#   lv0 = relax.call(rx.extern("relax.builtin.alloc_tensor"), (3, 4))
#   relax.call(rx.extern("test.op.identity"), (x, lv0))
#
#   return lv0


# After furthur lowering

# @rx.script
# def foo(x: Tensor[(3, 4), "float32"]):
#   gv0 = relax.call(extern("vm.builtin.alloc_storage"), (12, "cpu", "float32"))
#   gv1 = relax.call(extern("vm.builtin.alloc_tensor"), (gv0, 0, "float32", (3, 4)))
#   gv2 = relax.call(extern("test.op.identity"), (x, gv1))
#   relax.call(extern("vm.builtin.free_tensor"), (gv1))
#   relax.call(extern("vm.builtin.free_storage"), (gv0))


@tvm.register_func("test.op.identity")
def identity_packed(a, b):
    b = tvm.nd.array(a.asnumpy())

def Relax_to_VM():
    shape_anno = [3, 4]
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", shape_anno, type_anno)

    ib = rx.IRBuilder()

    storage_attr = tvm.ir.attrs.make_node(
        "relax.attrs.AllocStorageAttrs", device_id=0, device_type=1
    )
    tensor_attr = tvm.ir.attrs.make_node("relax.attrs.AllocTensorAttrs")

    with ib.function(x):
        gv0 = ib.emit(Call(rx.ExternFunc("vm.builtin.alloc_storage"),[rx.ShapeExpr([12]), rx.ShapeExpr([8])], storage_attr))
        gv1 = ib.emit(Call(rx.ExternFunc("vm.builtin.alloc_tensor"),[gv0, rx.ShapeExpr([0]), rx.ShapeExpr([3, 4])], tensor_attr))
        ib.emit(Call(rx.ExternFunc("test.op.identity"), [x, gv1]))
        ib.emit_output(gv1)
    expr = ib.get()

    exec = rx.transform.compile(expr)
    print(exec.astext())
    print(exec.aspython())

    input = tvm.nd.array(np.random.rand(3,4))
    vm = rx.VirtualMachine(exec, tvm.cpu())
    res = vm["main"](input)
    print(res)


if __name__ == "__main__":
    Relax_to_VM()
