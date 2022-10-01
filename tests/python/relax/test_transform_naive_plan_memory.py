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
from __future__ import annotations
import pytest
import tvm
from tvm import relax
from tvm import tir
from tvm.ir import assert_structural_equal


def _check(mod_before, mod_expected_mem_plan, mod_expected_vm_mem_lower):
    passes = [relax.transform.ToNonDataflow()]
    passes.append(relax.transform.CallTIRRewrite())
    passes.append(relax.transform.NaivePlanMemory())
    seq = tvm.transform.Sequential(passes)

    mod_after_mem_plan = seq(mod_before)
    assert_structural_equal(mod_after_mem_plan, mod_expected_mem_plan)

    mod_after_vm_mem_lower = relax.transform.VMMemoryLower()(mod_after_mem_plan)
    assert_structural_equal(mod_after_vm_mem_lower, mod_expected_vm_mem_lower)


m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
x = relax.Var("x", [m, n], relax.DynTensorType(2, "float32"))


# TODO(@lesheng): will add more testcases after blockCombine pass
def test_basic():
    def before():
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            y = bb.emit(relax.call_tir("foo", (x,), (m, n), dtype="float32"))
            bb.emit_func_output(y)

        return bb.get()

    def expected_mem_plan():
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            storage = bb.emit(
                relax.Call(
                    op=tvm.ir.Op.get("relax.memory.alloc_storage"),
                    args=[
                        relax.ShapeExpr(
                            [
                                (m * n) * 4,
                            ]
                        )
                    ],
                    attrs=tvm.ir.attrs.make_node(
                        "relax.attrs.MemAllocStorageAttrs",
                        virtual_device_index=0,
                        storage_scope="global",
                        dtype="float32",
                    ),
                )
            )
            alloc = bb.emit(
                relax.Call(
                    op=tvm.ir.Op.get("relax.memory.alloc_tensor"),
                    args=[
                        storage,
                        relax.ShapeExpr([m, n]),
                    ],
                    attrs=tvm.ir.attrs.make_node(
                        "relax.attrs.MemAllocTensorAttrs", offset=0, dtype="float32"
                    ),
                ),
            )
            _ = bb.emit(
                relax.Call(
                    op=relax.ExternFunc("foo"), args=[x, alloc], type_args=[relax.ObjectType()]
                )
            )
            y = bb.emit(alloc)
            bb.emit_func_output(y)

        return bb.get()

    def expected_vm_mem_lower():
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            storage = bb.emit(
                relax.Call(
                    op=tvm.ir.Op.get("relax.vm.builtin.alloc_storage"),
                    args=[relax.ShapeExpr([(m * n) * 4])],
                    attrs=tvm.ir.attrs.make_node(
                        "relax.attrs.VMAllocStorageAttrs", dtype="float32", runtime_device_index=0
                    ),
                )
            )
            alloc = bb.emit(
                relax.Call(
                    op=tvm.ir.Op.get("relax.vm.builtin.alloc_tensor"),
                    args=[storage, relax.ShapeExpr([m, n])],
                    attrs=tvm.ir.attrs.make_node(
                        "relax.attrs.VMAllocTensorAttrs", offset=0, dtype="float32"
                    ),
                )
            )
            _ = bb.emit(
                relax.Call(
                    op=relax.ExternFunc("foo"), args=[x, alloc], type_args=[relax.ObjectType()]
                )
            )
            y = bb.emit(alloc)
            bb.emit_func_output(y)

        return bb.get()

    _check(before(), expected_mem_plan(), expected_vm_mem_lower())


if __name__ == "__main__":
    pytest.main([__file__])
