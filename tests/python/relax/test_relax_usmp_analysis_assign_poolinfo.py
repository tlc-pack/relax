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

import sys
import pytest
import tvm
from tvm.ir import Span
from tvm.ir.memory_pools import WorkspaceMemoryPools
from tvm.relax import expr_functor, PyExprVisitor, PyExprMutator, Expr
from tvm.relax.testing import dump_ast
import tvm.script
import tvm.testing
from tvm import relax, rpc, te, tir, topi, TVMError, cpu, WorkspacePoolInfo, ConstantPoolInfo
from tvm.script import relax as R, tir as T
from tvm.target import Target


def _check_pool_infos(mod, target_to_pool_info):
    @relax.expr_functor.visitor
    class RelaxFuncCheck(PyExprVisitor):
        def visit_span(self, span: Span):
            pass

        def visit_call_(self, op: tvm.relax.Call):
            call = op
            if "relax.builtin.alloc_tensor" == str(call.op):
                candidate_pools = call.attrs["candidate_memory_pools"]
                assert candidate_pools[0] == pool_info
            return super().visit_call_(op)

    def check_poolinfos(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            assert stmt.annotations["candidate_memory_pools"][0] == pool_info
        return stmt

    for global_var, basefunc in mod.functions.items():
        pool_info = target_to_pool_info[basefunc.attrs["target"]]
        if isinstance(basefunc, tvm.relax.Function):
            RelaxFuncCheck().visit_expr(basefunc)
        if isinstance(basefunc, tvm.tir.PrimFunc):
            basefunc.with_body(
                tvm.tir.stmt_functor.ir_transform(basefunc.body, None, check_poolinfos)
            )


device = cpu(0)

# fmt: off
@tvm.script.ir_module
class RelaxAndTIR:
    @T.prim_func
    def prim_func_1(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "prim_func_1", "tir.noalias": True})
        placeholder_65 = T.match_buffer(placeholder_62, [150528], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_66 = T.match_buffer(placeholder_63, [9408], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_67 = T.match_buffer(placeholder_64, [64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_21 = T.match_buffer(T_cast_20, [289], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_7 = T.allocate([157323], "int16", "global")
        for i0_i1_fused_7 in T.serial(0, 229):
            for i2_7, i3_7 in T.grid(229, 3):
                PaddedInput_7[(((i0_i1_fused_7*687) + (i2_7*3)) + i3_7)] = T.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), placeholder_65[((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7 = T.allocate([64], "int32", "global")
            for ff_3 in T.serial(0, 64):
                Conv2dOutput_7[ff_3] = 0
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    Conv2dOutput_7[ff_3] = (Conv2dOutput_7[ff_3] + (T.cast(PaddedInput_7[(((((T.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (T.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)], "int32")*T.cast(placeholder_66[((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)], "int32")))
            for ax3_inner_7 in T.serial(0, 64):
                T_cast_21[((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_7[ax3_inner_7] + placeholder_67[ax3_inner_7]), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8")

    @R.function
    def __tvm_main__(input: Tensor((16, 16), "uint8")) -> Tensor:
        tsid_11 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_12 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")

        output = relax.call_tir("prim_func_1", (input, tsid_11, tsid_12), (802816, 1), dtype="int32")
        return output
# fmt: on


def test_relax_and_tir():
    target = Target("c")
    target_llvm = Target("llvm")
    relax_mod = RelaxAndTIR
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    relax_mod["__tvm_main__"] = relax_mod["__tvm_main__"].with_attr("target", target)
    relax_mod["prim_func_1"] = relax_mod["prim_func_1"].with_attr("target", target_llvm)

    c_target_pool = WorkspacePoolInfo(pool_name="c_target_pool", targets=[target])
    llvm_target_pool = WorkspacePoolInfo(pool_name="llvm_target_pool", targets=[target_llvm])
    workspace_memory_pools = WorkspaceMemoryPools([c_target_pool, llvm_target_pool])
    relax_mod = relax_mod.with_attr("workspace_memory_pools", workspace_memory_pools)

    relax_mod = tvm.relax.transform.AssignPoolInfo()(relax_mod)
    _check_pool_infos(relax_mod, {target: c_target_pool, target_llvm: llvm_target_pool})


# fmt: off
@tvm.script.ir_module
class RelaxAndTIRMultipleTargets:
    @T.prim_func
    def prim_func_1(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "prim_func_1", "tir.noalias": True})
        placeholder_65 = T.match_buffer(placeholder_62, [150528], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_66 = T.match_buffer(placeholder_63, [9408], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_67 = T.match_buffer(placeholder_64, [64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_21 = T.match_buffer(T_cast_20, [289], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_7 = T.allocate([157323], "int16", "global")
        for i0_i1_fused_7 in T.serial(0, 229):
            for i2_7, i3_7 in T.grid(229, 3):
                PaddedInput_7[(((i0_i1_fused_7*687) + (i2_7*3)) + i3_7)] = T.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), placeholder_65[((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7 = T.allocate([64], "int32", "global")
            for ff_3 in T.serial(0, 64):
                Conv2dOutput_7[ff_3] = 0
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    Conv2dOutput_7[ff_3] = (Conv2dOutput_7[ff_3] + (T.cast(PaddedInput_7[(((((T.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (T.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)], "int32")*T.cast(placeholder_66[((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)], "int32")))
            for ax3_inner_7 in T.serial(0, 64):
                T_cast_21[((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_7[ax3_inner_7] + placeholder_67[ax3_inner_7]), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def prim_func_2(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [177], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = T.allocate([200704], "uint8", "global")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)] = T.max(tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)], T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), placeholder_29[(((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)] = T.cast(tensor_2[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)], "int16")

    @R.function
    def __tvm_main__(input: Tensor((16, 16), "uint8")) -> Tensor:
        tsid_11 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_12 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")

        lv1 = relax.call_tir("prim_func_1", (input, tsid_11, tsid_12), (802816, 1), dtype="int32")
        output = relax.call_tir("prim_func_2", (lv1), (802816, 1), dtype="int32")
        return output
# fmt: on


def test_relax_and_tir_multiple_targets():
    target = Target("c")
    target_llvm = Target("llvm")
    target_cuda = Target("cuda")
    relax_mod = RelaxAndTIRMultipleTargets
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    relax_mod["__tvm_main__"] = relax_mod["__tvm_main__"].with_attr("target", target)
    relax_mod["prim_func_1"] = relax_mod["prim_func_1"].with_attr("target", target_llvm)
    relax_mod["prim_func_2"] = relax_mod["prim_func_2"].with_attr("target", target_cuda)

    c_target_pool = WorkspacePoolInfo(pool_name="c_target_pool", targets=[target])
    llvm_target_pool = WorkspacePoolInfo(pool_name="llvm_target_pool", targets=[target_llvm])
    cuda_target_pool = WorkspacePoolInfo(pool_name="cuda_target_pool", targets=[target_cuda])
    workspace_memory_pools = WorkspaceMemoryPools(
        [c_target_pool, llvm_target_pool, cuda_target_pool]
    )
    relax_mod = relax_mod.with_attr("workspace_memory_pools", workspace_memory_pools)

    relax_mod = tvm.relax.transform.AssignPoolInfo()(relax_mod)
    _check_pool_infos(
        relax_mod,
        {target: c_target_pool, target_llvm: llvm_target_pool, target_cuda: cuda_target_pool},
    )


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
