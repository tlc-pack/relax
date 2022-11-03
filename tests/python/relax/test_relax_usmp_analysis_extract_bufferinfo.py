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
from tvm.relax import expr_functor, PyExprVisitor, PyExprMutator, Expr
import tvm.script
import tvm.testing
from tvm import relax, rpc, te, tir, topi, TVMError, cpu, WorkspacePoolInfo, ConstantPoolInfo
from tvm.script import relax as R, tir as T
from tvm.target import Target


def _replace_stmt_with_buf_var_names(buffer_info_map):
    """helper to replace tir.allocates with buffer names"""
    new_buffer_info_map = dict()
    for k, v in buffer_info_map.items():
        new_buffer_info_map[k.name_hint] = k
    return new_buffer_info_map


def _verify_conflicts(main_buf_name, conflicting_buf_names, buffer_info_map):
    """helper to check expected liveness conflicts"""
    buf_info = buffer_info_map[main_buf_name]
    for conflict in buf_info.conflicts:
        assert conflict.name_hint in conflicting_buf_names


def _assign_poolinfos_to_allocates_in_primfuncs(func, pool_infos, constant_pool_infos):
    """helper to assing poolinfos to allocate nodes in a tir.PrimFunc"""

    def set_poolinfos(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            return tvm.tir.Allocate(
                buffer_var=stmt.buffer_var,
                dtype=stmt.dtype,
                extents=stmt.extents,
                condition=stmt.condition,
                body=stmt.body,
                annotations={tvm.tir.usmp.utils.CANDIDATE_MEMORY_POOL_ATTR: pool_infos},
            )
        elif isinstance(stmt, tvm.tir.AllocateConst):
            return tvm.tir.AllocateConst(
                buffer_var=stmt.buffer_var,
                dtype=stmt.dtype,
                extents=stmt.extents,
                data_or_idx=stmt.data,
                body=stmt.body,
                annotations={tvm.tir.usmp.utils.CANDIDATE_MEMORY_POOL_ATTR: constant_pool_infos},
            )

    return func.with_body(tvm.tir.stmt_functor.ir_transform(func.body, None, set_poolinfos))


def _assign_poolinfos_to_allocates_in_irmodule(mod, pool_infos, constant_pool_infos=None):
    """helper to assign poolinfos to allocate nodes in a IRModule"""

    @relax.expr_functor.mutator
    class RelaxFuncAnnotate(PyExprMutator):
        def visit_span(self, span: Span) -> Span:
            pass

        def visit_call_(self, op: tvm.relax.Call) -> Expr:
            call = op
            if "relax.builtin.alloc_tensor" == str(call.op):
                attrs = tvm.ir.attrs.make_node(
                    "relax.attrs.AllocTensorAttrs",
                    dtype=call.attrs["dtype"],
                    runtime_device_index=call.attrs["runtime_device_index"],
                    candidate_memory_pools=pool_infos,
                )
                return tvm.relax.Call(call.op, call.args, attrs, call.type_args, call.span)
            return super().visit_call_(op)

    relax_visitor = RelaxFuncAnnotate()
    mod["run_model"] = relax_visitor.visit_expr(mod["run_model"])

    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, tvm.tir.PrimFunc):
            ret[global_var] = _assign_poolinfos_to_allocates_in_primfuncs(
                basefunc, pool_infos, constant_pool_infos
            )
        else:
            ret[global_var] = basefunc
    return ret


def _assign_targets_to_relaxfuncs_irmodule(mod, target):
    """helper to assign target for PrimFunc in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, (tvm.relax.Function, tvm.tir.PrimFunc)):
            ret[global_var] = basefunc.with_attr("target", target)
    return ret


device = cpu(0)

# fmt: off
@tvm.script.ir_module
class LinearStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "tir.noalias": True})
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
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
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
    def run_model(input: Tensor((16, 16), "uint8")) -> Tensor:
        tsid_10 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_11 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_12 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")

        lv0 = relax.call_tir("tvmgen_default_fused_cast_subtract", (input, tsid_10), (301056, 1), dtype="int32")
        lv1 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", (lv0, tsid_11, tsid_12), (802816, 1), dtype="int32")
        output = relax.call_tir("tvmgen_default_fused_nn_max_pool2d_cast", (lv1), (16, 16), dtype="int32")
        return output
# fmt: on


def test_linear():
    target = Target("c")
    relax_mod = LinearStructure
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    fast_memory_pool = WorkspacePoolInfo(pool_name="fast_memory", targets=[target])
    slow_memory_pool = WorkspacePoolInfo(pool_name="slow_memory", targets=[target])

    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)
    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(
        relax_mod, [fast_memory_pool, slow_memory_pool]
    )

    buffer_info_analysis = tvm.relax.analysis.extract_buffer_info(relax_mod["run_model"], relax_mod)
    buffer_info_map_relax = _replace_stmt_with_buf_var_names(buffer_info_analysis.buffer_info_stmts)

    assert buffer_info_analysis.memory_pressure == 3526168

    # check conflicts
    _verify_conflicts(
        "PaddedInput_7",
        ["alloc", "Conv2dOutput_7", "alloc1", "tsid_11", "tsid_12"],
        buffer_info_map_relax,
    )
    _verify_conflicts("tsid_10", ["alloc"], buffer_info_map_relax)
    _verify_conflicts("alloc2", ["tensor_2"], buffer_info_map_relax)
    _verify_conflicts("alloc", ["tsid_10", "PaddedInput_7"], buffer_info_map_relax)
    _verify_conflicts(
        "tsid_11", ["PaddedInput_7", "alloc1", "tsid_12", "Conv2dOutput_7"], buffer_info_map_relax
    )
    _verify_conflicts(
        "alloc1",
        ["tsid_11", "PaddedInput_7", "tsid_12", "Conv2dOutput_7", "tensor_2"],
        buffer_info_map_relax,
    )
    _verify_conflicts(
        "tsid_12", ["alloc1", "PaddedInput_7", "tsid_11", "Conv2dOutput_7"], buffer_info_map_relax
    )
    _verify_conflicts(
        "Conv2dOutput_7", ["tsid_12", "alloc1", "PaddedInput_7", "tsid_11"], buffer_info_map_relax
    )
    _verify_conflicts("tensor_2", ["alloc1", "alloc2"], buffer_info_map_relax)

    # check sizes
    assert buffer_info_map_relax["alloc"].size_bytes == 1204224
    assert buffer_info_map_relax["alloc1"].size_bytes == 3211264
    assert buffer_info_map_relax["alloc2"].size_bytes == 1024
    assert buffer_info_map_relax["Conv2dOutput_7"].size_bytes == 256
    assert buffer_info_map_relax["PaddedInput_7"].size_bytes == 314646
    assert buffer_info_map_relax["tensor_2"].size_bytes == 200704

    # check_pool_candidates
    assert [
        pool_info.pool_name for pool_info in list(buffer_info_map_relax["alloc"].pool_candidates)
    ] == ["fast_memory", "slow_memory"]


# fmt: off
@tvm.script.ir_module
class ParallelSerialMixedForLoops:
    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1(placeholder_68: T.handle, placeholder_69: T.handle, placeholder_70: T.handle, T_cast_22: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", "tir.noalias": True})
        placeholder_71 = T.match_buffer(placeholder_68, [262144], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_72 = T.match_buffer(placeholder_69, [110592], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_73 = T.match_buffer(placeholder_70, [192], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_23 = T.match_buffer(T_cast_22, [305], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_8 = T.allocate([215296], "int16", "global")
        for i0_i1_fused_8 in T.serial(0, 58):
            for i2_8, i3_8 in T.grid(58, 64):
                PaddedInput_8[(((i0_i1_fused_8*3712) + (i2_8*64)) + i3_8)] = T.if_then_else(((((1 <= i0_i1_fused_8) and (i0_i1_fused_8 < 57)) and (1 <= i2_8)) and (i2_8 < 57)), placeholder_71[((((i0_i1_fused_8*3584) + (i2_8*64)) + i3_8) - 3648)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_8 in T.parallel(0, 3136):
            dummy_allocate = T.allocate([1], "int32", "global")
            for ax3_outer_4 in T.serial(0, 3):
                Conv2dOutput_8 = T.allocate([64], "int32", "global")
                for ff_4 in T.serial(0, 64):
                    Conv2dOutput_8[ff_4] = 0
                    for ry_3, rx_3, rc_8 in T.grid(3, 3, 64):
                        Conv2dOutput_8[ff_4] = (Conv2dOutput_8[ff_4] + (T.cast(PaddedInput_8[(((((T.floordiv(ax0_ax1_fused_ax2_fused_8, 56)*3712) + (ry_3*3712)) + (rx_3*64)) + (T.floormod(ax0_ax1_fused_ax2_fused_8, 56)*64)) + rc_8)], "int32")*T.cast(placeholder_72[(((((ry_3*36864) + (rx_3*12288)) + (rc_8*192)) + (ax3_outer_4*64)) + ff_4)], "int32")))
                for ax3_inner_8 in T.serial(0, 64):
                    T_cast_23[(((ax0_ax1_fused_ax2_fused_8*192) + (ax3_outer_4*64)) + ax3_inner_8)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_8[ax3_inner_8] + placeholder_73[((ax3_outer_4*64) + ax3_inner_8)]), 1139793473, 31, -6, dtype="int32"), 255), 0), "uint8")

    @R.function
    def run_model(input: Tensor((512, 512), "uint8")) -> Tensor:
        tsid_10 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_11 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")

        output = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", (input, tsid_10, tsid_11), (262144, 1), dtype="int32")
        return output
# fmt: on


# fmt: off
@tvm.script.ir_module
class AllSerialForLoops:
    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1(placeholder_68: T.handle, placeholder_69: T.handle, placeholder_70: T.handle, T_cast_22: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", "tir.noalias": True})
        placeholder_71 = T.match_buffer(placeholder_68, [200704], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_72 = T.match_buffer(placeholder_69, [110592], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_73 = T.match_buffer(placeholder_70, [192], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_23 = T.match_buffer(T_cast_22, [305], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_8 = T.allocate([215296], "int16", "global")
        for i0_i1_fused_8 in T.serial(0, 58):
            for i2_8, i3_8 in T.grid(58, 64):
                PaddedInput_8[(((i0_i1_fused_8*3712) + (i2_8*64)) + i3_8)] = T.if_then_else(((((1 <= i0_i1_fused_8) and (i0_i1_fused_8 < 57)) and (1 <= i2_8)) and (i2_8 < 57)), placeholder_71[((((i0_i1_fused_8*3584) + (i2_8*64)) + i3_8) - 3648)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_8 in T.serial(0, 3136):
            dummy_allocate = T.allocate([1], "int32", "global")
            for ax3_outer_4 in T.serial(0, 3):
                Conv2dOutput_8 = T.allocate([64], "int32", "global")
                for ff_4 in T.serial(0, 64):
                    Conv2dOutput_8[ff_4] = 0
                    for ry_3, rx_3, rc_8 in T.grid(3, 3, 64):
                        Conv2dOutput_8[ff_4] = (Conv2dOutput_8[ff_4] + (T.cast(PaddedInput_8[(((((T.floordiv(ax0_ax1_fused_ax2_fused_8, 56)*3712) + (ry_3*3712)) + (rx_3*64)) + (T.floormod(ax0_ax1_fused_ax2_fused_8, 56)*64)) + rc_8)], "int32")*T.cast(placeholder_72[(((((ry_3*36864) + (rx_3*12288)) + (rc_8*192)) + (ax3_outer_4*64)) + ff_4)], "int32")))
                for ax3_inner_8 in T.serial(0, 64):
                    T_cast_23[(((ax0_ax1_fused_ax2_fused_8*192) + (ax3_outer_4*64)) + ax3_inner_8)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_8[ax3_inner_8] + placeholder_73[((ax3_outer_4*64) + ax3_inner_8)]), 1139793473, 31, -6, dtype="int32"), 255), 0), "uint8")


    @R.function
    def run_model(input: Tensor((512, 512), "uint8")) -> Tensor:
        tsid_10 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_11 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")

        output = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", (input, tsid_10, tsid_11), (262144, 1), dtype="int32")
        return output
# fmt: on


def test_parallel_serial_mixed_for_loops():
    target = Target("c")
    global_ws_pool = WorkspacePoolInfo(
        pool_name="global_workspace",
        targets=[target],
    )
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)

    all_serial_mod = AllSerialForLoops
    all_serial_mod = seq(all_serial_mod)
    all_serial_mod = _assign_targets_to_relaxfuncs_irmodule(all_serial_mod, target)
    all_serial_mod = _assign_poolinfos_to_allocates_in_irmodule(all_serial_mod, [global_ws_pool])
    main_func = all_serial_mod["run_model"]
    buffer_info_analysis = tvm.relax.analysis.extract_buffer_info(main_func, all_serial_mod)
    assert buffer_info_analysis.memory_pressure == 1479426
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_analysis.buffer_info_stmts)

    # When all loops are serial all allocates are touched by USMP
    assert len(buffer_info_map) == 6
    for name, _ in buffer_info_map.items():
        assert name in [
            "alloc",
            "tsid_10",
            "tsid_11",
            "dummy_allocate",
            "Conv2dOutput_8",
            "PaddedInput_8",
        ]

    parallel_serial_mixed_tir_mod = ParallelSerialMixedForLoops
    parallel_serial_mixed_tir_mod = seq(parallel_serial_mixed_tir_mod)
    parallel_serial_mixed_tir_mod = _assign_targets_to_relaxfuncs_irmodule(
        parallel_serial_mixed_tir_mod, target
    )
    parallel_serial_mixed_tir_mod = _assign_poolinfos_to_allocates_in_irmodule(
        parallel_serial_mixed_tir_mod, [global_ws_pool]
    )
    main_func = parallel_serial_mixed_tir_mod["run_model"]
    buffer_info_analysis = tvm.relax.analysis.extract_buffer_info(
        main_func, parallel_serial_mixed_tir_mod
    )
    assert buffer_info_analysis.memory_pressure == 1479426
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_analysis.buffer_info_stmts)

    # USMP will not touch (yet) the allocates inside parallel for loops
    assert len(buffer_info_map) == 5
    for name, _ in buffer_info_map.items():
        assert name in ["alloc", "tsid_10", "tsid_11", "Conv2dOutput_8", "PaddedInput_8"]


# fmt: off
@tvm.script.ir_module
class InceptionStructure:
    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d(placeholder: T.handle, tensor: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d", "tir.noalias": True})
        placeholder_1 = T.match_buffer(placeholder, [602112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        tensor_1 = T.match_buffer(tensor, [249], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused in T.serial(0, 28):
            for ax2 in T.serial(0, 28):
                for ax3_outer_init, ax3_inner_init in T.grid(3, 64):
                    tensor_1[((((ax0_ax1_fused*5376) + (ax2*192)) + (ax3_outer_init*64)) + ax3_inner_init)] = T.uint8(0)
                for rv0_rv1_fused, ax3_outer, ax3_inner in T.grid(9, 3, 64):
                    tensor_1[((((ax0_ax1_fused*5376) + (ax2*192)) + (ax3_outer*64)) + ax3_inner)] = T.max(tensor_1[((((ax0_ax1_fused*5376) + (ax2*192)) + (ax3_outer*64)) + ax3_inner)], T.if_then_else(((((ax0_ax1_fused*2) + T.floordiv(rv0_rv1_fused, 3)) < 56) and (((ax2*2) + T.floormod(rv0_rv1_fused, 3)) < 56)), placeholder_1[((((((ax0_ax1_fused*21504) + (T.floordiv(rv0_rv1_fused, 3)*10752)) + (ax2*384)) + (T.floormod(rv0_rv1_fused, 3)*192)) + (ax3_outer*64)) + ax3_inner)], T.uint8(0), dtype="uint8"))

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def tvmgen_default_fused_cast(placeholder_6: T.handle, T_cast: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast", "tir.noalias": True})
        placeholder_7 = T.match_buffer(placeholder_6, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_1 = T.match_buffer(T_cast, [249], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_2 in T.serial(0, 28):
            for ax2_2, ax3_outer_1, ax3_inner_2 in T.grid(28, 12, 16):
                T_cast_1[((((ax0_ax1_fused_2*5376) + (ax2_2*192)) + (ax3_outer_1*16)) + ax3_inner_2)] = T.cast(placeholder_7[((((ax0_ax1_fused_2*5376) + (ax2_2*192)) + (ax3_outer_1*16)) + ax3_inner_2)], "int16")

    @T.prim_func
    def tvmgen_default_fused_concatenate(placeholder_8: T.handle, placeholder_9: T.handle, placeholder_10: T.handle, placeholder_11: T.handle, T_concat: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_concatenate", "tir.noalias": True})
        placeholder_12 = T.match_buffer(placeholder_8, [50176], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_concat_1 = T.match_buffer(T_concat, [313], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_13 = T.match_buffer(placeholder_9, [100352], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_14 = T.match_buffer(placeholder_11, [25088], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_15 = T.match_buffer(placeholder_10, [25088], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_3 in T.serial(0, 28):
            for ax2_3, ax3 in T.grid(28, 256):
                T_concat_1[(((ax0_ax1_fused_3*7168) + (ax2_3*256)) + ax3)] = T.if_then_else((224 <= ax3), placeholder_14[((((ax0_ax1_fused_3*896) + (ax2_3*32)) + ax3) - 224)], T.if_then_else((192 <= ax3), placeholder_15[((((ax0_ax1_fused_3*896) + (ax2_3*32)) + ax3) - 192)], T.if_then_else((64 <= ax3), placeholder_13[((((ax0_ax1_fused_3*3584) + (ax2_3*128)) + ax3) - 64)], placeholder_12[(((ax0_ax1_fused_3*1792) + (ax2_3*64)) + ax3)], dtype="uint8"), dtype="uint8"), dtype="uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_cast_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", "tir.noalias": True})
        placeholder_19 = T.match_buffer(placeholder_16, [200704], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_20 = T.match_buffer(placeholder_17, [4096], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_21 = T.match_buffer(placeholder_18, [64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_3 = T.match_buffer(T_cast_2, [177], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput = T.allocate([200704], "int16", "global")
        for i0_i1_fused in T.serial(0, 56):
            for i2, i3 in T.grid(56, 64):
                PaddedInput[(((i0_i1_fused*3584) + (i2*64)) + i3)] = placeholder_19[(((i0_i1_fused*3584) + (i2*64)) + i3)]
        for ax0_ax1_fused_ax2_fused in T.serial(0, 3136):
            Conv2dOutput = T.allocate([64], "int32", "global")
            for ff in T.serial(0, 64):
                Conv2dOutput[ff] = 0
                for rc in T.serial(0, 64):
                    Conv2dOutput[ff] = (Conv2dOutput[ff] + (T.cast(PaddedInput[((ax0_ax1_fused_ax2_fused*64) + rc)], "int32")*T.cast(placeholder_20[((rc*64) + ff)], "int32")))
            for ax3_inner_3 in T.serial(0, 64):
                T_cast_3[((ax0_ax1_fused_ax2_fused*64) + ax3_inner_3)] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput[ax3_inner_3] + placeholder_21[ax3_inner_3]), 1191576922, 31, -4, dtype="int32"), 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, T_cast_4: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", "tir.noalias": True})
        placeholder_25 = T.match_buffer(placeholder_22, [150528], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_26 = T.match_buffer(placeholder_23, [18432], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_27 = T.match_buffer(placeholder_24, [96], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_5 = T.match_buffer(T_cast_4, [153], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_1 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_1 in T.serial(0, 28):
            for i2_1, i3_1 in T.grid(28, 192):
                PaddedInput_1[(((i0_i1_fused_1*5376) + (i2_1*192)) + i3_1)] = placeholder_25[(((i0_i1_fused_1*5376) + (i2_1*192)) + i3_1)]
        for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 784):
            Conv2dOutput_1 = T.allocate([1], "int32", "global")
            for ax3_1 in T.serial(0, 96):
                Conv2dOutput_1[0] = 0
                for rc_1 in T.serial(0, 192):
                    Conv2dOutput_1[0] = (Conv2dOutput_1[0] + (T.cast(PaddedInput_1[((ax0_ax1_fused_ax2_fused_1*192) + rc_1)], "int32")*T.cast(placeholder_26[((rc_1*96) + ax3_1)], "int32")))
                T_cast_5[((ax0_ax1_fused_ax2_fused_1*96) + ax3_1)] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_1[0] + placeholder_27[ax3_1]), 1201322342, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
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

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_2(placeholder_30: T.handle, placeholder_31: T.handle, placeholder_32: T.handle, T_cast_8: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_2", "tir.noalias": True})
        placeholder_33 = T.match_buffer(placeholder_30, [150528], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_34 = T.match_buffer(placeholder_31, [12288], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_35 = T.match_buffer(placeholder_32, [64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_9 = T.match_buffer(T_cast_8, [121], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_2 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_2 in T.serial(0, 28):
            for i2_2, i3_2 in T.grid(28, 192):
                PaddedInput_2[(((i0_i1_fused_2*5376) + (i2_2*192)) + i3_2)] = placeholder_33[(((i0_i1_fused_2*5376) + (i2_2*192)) + i3_2)]
        for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 784):
            Conv2dOutput_2 = T.allocate([64], "int32", "global")
            for ff_1 in T.serial(0, 64):
                Conv2dOutput_2[ff_1] = 0
                for rc_2 in T.serial(0, 192):
                    Conv2dOutput_2[ff_1] = (Conv2dOutput_2[ff_1] + (T.cast(PaddedInput_2[((ax0_ax1_fused_ax2_fused_2*192) + rc_2)], "int32")*T.cast(placeholder_34[((rc_2*64) + ff_1)], "int32")))
            for ax3_inner_4 in T.serial(0, 64):
                T_cast_9[((ax0_ax1_fused_ax2_fused_2*64) + ax3_inner_4)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_2[ax3_inner_4] + placeholder_35[ax3_inner_4]), 1663316467, 31, -7, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast_1(placeholder_36: T.handle, T_cast_10: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast_1", "tir.noalias": True})
        placeholder_37 = T.match_buffer(placeholder_36, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_11 = T.match_buffer(T_cast_10, [249], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_3 = T.allocate([150528], "uint8", "global")
        for ax0_ax1_fused_6 in T.serial(0, 28):
            for ax2_6 in T.serial(0, 28):
                for ax3_outer_init_1, ax3_inner_init_1 in T.grid(3, 64):
                    tensor_3[((((ax0_ax1_fused_6*5376) + (ax2_6*192)) + (ax3_outer_init_1*64)) + ax3_inner_init_1)] = T.uint8(0)
                for rv0_rv1_fused_2, ax3_outer_2, ax3_inner_5 in T.grid(9, 3, 64):
                    tensor_3[((((ax0_ax1_fused_6*5376) + (ax2_6*192)) + (ax3_outer_2*64)) + ax3_inner_5)] = T.max(tensor_3[((((ax0_ax1_fused_6*5376) + (ax2_6*192)) + (ax3_outer_2*64)) + ax3_inner_5)], T.if_then_else(((((1 <= (T.floordiv(rv0_rv1_fused_2, 3) + ax0_ax1_fused_6)) and ((T.floordiv(rv0_rv1_fused_2, 3) + ax0_ax1_fused_6) < 29)) and (1 <= (ax2_6 + T.floormod(rv0_rv1_fused_2, 3)))) and ((ax2_6 + T.floormod(rv0_rv1_fused_2, 3)) < 29)), placeholder_37[(((((((T.floordiv(rv0_rv1_fused_2, 3)*5376) + (ax0_ax1_fused_6*5376)) + (ax2_6*192)) + (T.floormod(rv0_rv1_fused_2, 3)*192)) + (ax3_outer_2*64)) + ax3_inner_5) - 5568)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_7 in T.serial(0, 28):
            for ax2_7, ax3_4 in T.grid(28, 192):
                T_cast_11[(((ax0_ax1_fused_7*5376) + (ax2_7*192)) + ax3_4)] = T.cast(tensor_3[(((ax0_ax1_fused_7*5376) + (ax2_7*192)) + ax3_4)], "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__2(placeholder_38: T.handle, placeholder_39: T.handle, placeholder_40: T.handle, T_cast_12: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__2", "tir.noalias": True})
        placeholder_41 = T.match_buffer(placeholder_38, [150528], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_42 = T.match_buffer(placeholder_39, [6144], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_43 = T.match_buffer(placeholder_40, [32], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_13 = T.match_buffer(T_cast_12, [89], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_3 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_3 in T.serial(0, 28):
            for i2_3, i3_3 in T.grid(28, 192):
                PaddedInput_3[(((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3)] = placeholder_41[(((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3)]
        for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 784):
            Conv2dOutput_3 = T.allocate([1], "int32", "global")
            for ax3_5 in T.serial(0, 32):
                Conv2dOutput_3[0] = 0
                for rc_3 in T.serial(0, 192):
                    Conv2dOutput_3[0] = (Conv2dOutput_3[0] + (T.cast(PaddedInput_3[((ax0_ax1_fused_ax2_fused_3*192) + rc_3)], "int32")*T.cast(placeholder_42[((rc_3*32) + ax3_5)], "int32")))
                T_cast_13[((ax0_ax1_fused_ax2_fused_3*32) + ax3_5)] = T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_3[0] + placeholder_43[ax3_5]), 1811141736, 31, -6, dtype="int32"), 255), 0), "uint8"), "int32"), 1136333842, 31, 0, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2(placeholder_44: T.handle, placeholder_45: T.handle, placeholder_46: T.handle, T_cast_14: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2", "tir.noalias": True})
        placeholder_47 = T.match_buffer(placeholder_44, [150528], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_48 = T.match_buffer(placeholder_45, [3072], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_49 = T.match_buffer(placeholder_46, [16], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_15 = T.match_buffer(T_cast_14, [73], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_4 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_4 in T.serial(0, 28):
            for i2_4, i3_4 in T.grid(28, 192):
                PaddedInput_4[(((i0_i1_fused_4*5376) + (i2_4*192)) + i3_4)] = placeholder_47[(((i0_i1_fused_4*5376) + (i2_4*192)) + i3_4)]
        for ax0_ax1_fused_ax2_fused_4 in T.serial(0, 784):
            Conv2dOutput_4 = T.allocate([1], "int32", "global")
            for ax3_6 in T.serial(0, 16):
                Conv2dOutput_4[0] = 0
                for rc_4 in T.serial(0, 192):
                    Conv2dOutput_4[0] = (Conv2dOutput_4[0] + (T.cast(PaddedInput_4[((ax0_ax1_fused_ax2_fused_4*192) + rc_4)], "int32")*T.cast(placeholder_48[((rc_4*16) + ax3_6)], "int32")))
                T_cast_15[((ax0_ax1_fused_ax2_fused_4*16) + ax3_6)] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_4[0] + placeholder_49[ax3_6]), 1764006585, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__1(placeholder_50: T.handle, placeholder_51: T.handle, placeholder_52: T.handle, T_cast_16: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__1", "tir.noalias": True})
        placeholder_53 = T.match_buffer(placeholder_50, [12544], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_54 = T.match_buffer(placeholder_51, [4608], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_55 = T.match_buffer(placeholder_52, [32], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_17 = T.match_buffer(T_cast_16, [89], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_5 = T.allocate([14400], "int16", "global")
        for i0_i1_fused_5 in T.serial(0, 30):
            for i2_5, i3_5 in T.grid(30, 16):
                PaddedInput_5[(((i0_i1_fused_5*480) + (i2_5*16)) + i3_5)] = T.if_then_else(((((1 <= i0_i1_fused_5) and (i0_i1_fused_5 < 29)) and (1 <= i2_5)) and (i2_5 < 29)), placeholder_53[((((i0_i1_fused_5*448) + (i2_5*16)) + i3_5) - 464)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_5 in T.serial(0, 784):
            Conv2dOutput_5 = T.allocate([1], "int32", "global")
            for ax3_7 in T.serial(0, 32):
                Conv2dOutput_5[0] = 0
                for ry, rx, rc_5 in T.grid(3, 3, 16):
                    Conv2dOutput_5[0] = (Conv2dOutput_5[0] + (T.cast(PaddedInput_5[(((((T.floordiv(ax0_ax1_fused_ax2_fused_5, 28)*480) + (ry*480)) + (rx*16)) + (T.floormod(ax0_ax1_fused_ax2_fused_5, 28)*16)) + rc_5)], "int32")*T.cast(placeholder_54[((((ry*1536) + (rx*512)) + (rc_5*32)) + ax3_7)], "int32")))
                T_cast_17[((ax0_ax1_fused_ax2_fused_5*32) + ax3_7)] = T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_5[0] + placeholder_55[ax3_7]), 1131968888, 31, -6, dtype="int32"), 255), 0), "uint8"), "int32"), 1900719667, 31, 0, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320_(placeholder_56: T.handle, placeholder_57: T.handle, placeholder_58: T.handle, T_cast_18: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320_", "tir.noalias": True})
        placeholder_59 = T.match_buffer(placeholder_56, [75264], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_60 = T.match_buffer(placeholder_57, [110592], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_61 = T.match_buffer(placeholder_58, [128], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_19 = T.match_buffer(T_cast_18, [185], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_6 = T.allocate([86400], "int16", "global")
        for i0_i1_fused_6 in T.serial(0, 30):
            for i2_6, i3_6 in T.grid(30, 96):
                PaddedInput_6[(((i0_i1_fused_6*2880) + (i2_6*96)) + i3_6)] = T.if_then_else(((((1 <= i0_i1_fused_6) and (i0_i1_fused_6 < 29)) and (1 <= i2_6)) and (i2_6 < 29)), placeholder_59[((((i0_i1_fused_6*2688) + (i2_6*96)) + i3_6) - 2784)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_6 in T.serial(0, 784):
            Conv2dOutput_6 = T.allocate([64], "int32", "global")
            for ax3_outer_3 in T.serial(0, 2):
                for ff_2 in T.serial(0, 64):
                    Conv2dOutput_6[ff_2] = 0
                    for ry_1, rx_1, rc_6 in T.grid(3, 3, 96):
                        Conv2dOutput_6[ff_2] = (Conv2dOutput_6[ff_2] + (T.cast(PaddedInput_6[(((((T.floordiv(ax0_ax1_fused_ax2_fused_6, 28)*2880) + (ry_1*2880)) + (rx_1*96)) + (T.floormod(ax0_ax1_fused_ax2_fused_6, 28)*96)) + rc_6)], "int32")*T.cast(placeholder_60[(((((ry_1*36864) + (rx_1*12288)) + (rc_6*128)) + (ax3_outer_3*64)) + ff_2)], "int32")))
                for ax3_inner_6 in T.serial(0, 64):
                    T_cast_19[(((ax0_ax1_fused_ax2_fused_6*128) + (ax3_outer_3*64)) + ax3_inner_6)] = T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_6[ax3_inner_6] + placeholder_61[((ax3_outer_3*64) + ax3_inner_6)]), 1374050734, 31, -7, dtype="int32"), 255), 0), "uint8"), "int32"), 1544713713, 31, 0, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "T.noalias": True})
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
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1(placeholder_68: T.handle, placeholder_69: T.handle, placeholder_70: T.handle, T_cast_22: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", "tir.noalias": True})
        placeholder_71 = T.match_buffer(placeholder_68, [200704], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_72 = T.match_buffer(placeholder_69, [110592], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_73 = T.match_buffer(placeholder_70, [192], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_23 = T.match_buffer(T_cast_22, [305], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_8 = T.allocate([215296], "int16", "global")
        for i0_i1_fused_8 in T.serial(0, 58):
            for i2_8, i3_8 in T.grid(58, 64):
                PaddedInput_8[(((i0_i1_fused_8*3712) + (i2_8*64)) + i3_8)] = T.if_then_else(((((1 <= i0_i1_fused_8) and (i0_i1_fused_8 < 57)) and (1 <= i2_8)) and (i2_8 < 57)), placeholder_71[((((i0_i1_fused_8*3584) + (i2_8*64)) + i3_8) - 3648)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_8 in T.serial(0, 3136):
            Conv2dOutput_8 = T.allocate([64], "int32", "global")
            for ax3_outer_4 in T.serial(0, 3):
                for ff_4 in T.serial(0, 64):
                    Conv2dOutput_8[ff_4] = 0
                    for ry_3, rx_3, rc_8 in T.grid(3, 3, 64):
                        Conv2dOutput_8[ff_4] = (Conv2dOutput_8[ff_4] + (T.cast(PaddedInput_8[(((((T.floordiv(ax0_ax1_fused_ax2_fused_8, 56)*3712) + (ry_3*3712)) + (rx_3*64)) + (T.floormod(ax0_ax1_fused_ax2_fused_8, 56)*64)) + rc_8)], "int32")*T.cast(placeholder_72[(((((ry_3*36864) + (rx_3*12288)) + (rc_8*192)) + (ax3_outer_4*64)) + ff_4)], "int32")))
                for ax3_inner_8 in T.serial(0, 64):
                    T_cast_23[(((ax0_ax1_fused_ax2_fused_8*192) + (ax3_outer_4*64)) + ax3_inner_8)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_8[ax3_inner_8] + placeholder_73[((ax3_outer_4*64) + ax3_inner_8)]), 1139793473, 31, -6, dtype="int32"), 255), 0), "uint8")

    @R.function
    def run_model(input: Tensor((16, 16), "uint8")) -> Tensor:
        tsid_100 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_101 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        sid_9 = relax.call_tir("tvmgen_default_fused_cast_subtract", (input, tsid_100), (301056, 1), dtype="int32")
        sid_8 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", (sid_9, tsid_100, tsid_101), (802816, 1), dtype="int32")
        sid_7 = relax.call_tir("tvmgen_default_fused_nn_max_pool2d_cast", (sid_8), (401408, 1), dtype="int32")
        sid_6 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", (sid_7, tsid_100, tsid_101), (401408, 1), dtype="int32")
        sid_5 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", (sid_6, tsid_100, tsid_101), (602112, 1), dtype="int32")
        sid_4 = relax.call_tir("tvmgen_default_fused_nn_max_pool2d", (sid_5), (150528, 1), dtype="int32")
        sid_3 = relax.call_tir("tvmgen_default_fused_cast", (sid_4), (301056, 1), dtype="int32")
        sid_2 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_2", (sid_3, tsid_100, tsid_101), (50176, 1), dtype="int32")
        sid_20 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", (sid_3, tsid_100, tsid_101), (150528, 1), dtype="int32")
        sid_19 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320_", (sid_20, tsid_100, tsid_101), (100352, 1), dtype="int32")
        sid_26 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2", (sid_3, tsid_100, tsid_101), (25088, 1), dtype="int32")
        sid_25 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__1", (sid_26, tsid_100, tsid_101), (25088, 1), dtype="int32")
        sid_32 = relax.call_tir("tvmgen_default_fused_nn_max_pool2d_cast_1", (sid_4), (301056, 1), dtype="int32")
        sid_31 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__2", (sid_32, tsid_100, tsid_101), (25088, 1), dtype="int32")
        output = relax.call_tir("tvmgen_default_fused_concatenate", (sid_2, sid_19, sid_25, sid_31), (25088, 1), dtype="int32")
        return output
# fmt: on


def test_inception_structure():
    target = Target("c")
    global_ws_pool = WorkspacePoolInfo(
        pool_name="global_workspace",
        targets=[target],
    )
    relax_mod = InceptionStructure
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)
    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)
    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(relax_mod, [global_ws_pool])
    main_func = relax_mod["run_model"]
    buffer_info_analysis = tvm.relax.analysis.extract_buffer_info(main_func, relax_mod)
    assert buffer_info_analysis.memory_pressure == 3526168
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_analysis.buffer_info_stmts)

    # check conflicts
    _verify_conflicts(
        "alloc9",
        [
            "PaddedInput_6",
            "alloc7",
            "tsid_100",
            "tsid_101",
            "alloc5",
            "alloc6",
            "Conv2dOutput_6",
            "PaddedInput_4",
            "alloc10",
            "Conv2dOutput_4",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "alloc11",
            "tensor_3",
            "alloc12",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "alloc13",
            "alloc14",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tensor_3",
        [
            "alloc11",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc5",
            "alloc12",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc1",
        [
            "Conv2dOutput_7",
            "PaddedInput_7",
            "tsid_100",
            "tsid_101",
            "tensor_2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_4",
        [
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc5",
            "alloc6",
            "alloc10",
            "Conv2dOutput_4",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_8",
        [
            "alloc3",
            "tsid_101",
            "tsid_100",
            "Conv2dOutput_8",
            "alloc4",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc8",
        [
            "PaddedInput_1",
            "alloc7",
            "tsid_100",
            "tsid_101",
            "alloc5",
            "alloc6",
            "Conv2dOutput_1",
            "PaddedInput_6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tsid_101",
        [
            "alloc1",
            "Conv2dOutput_7",
            "PaddedInput_7",
            "tsid_100",
            "tensor_2",
            "alloc2",
            "PaddedInput",
            "Conv2dOutput",
            "alloc3",
            "PaddedInput_8",
            "Conv2dOutput_8",
            "alloc4",
            "alloc5",
            "alloc6",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "alloc7",
            "PaddedInput_1",
            "alloc8",
            "Conv2dOutput_1",
            "PaddedInput_6",
            "alloc9",
            "Conv2dOutput_6",
            "PaddedInput_4",
            "alloc10",
            "Conv2dOutput_4",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "alloc11",
            "tensor_3",
            "alloc12",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "alloc13",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput",
        [
            "PaddedInput",
            "tsid_101",
            "tsid_100",
            "alloc3",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_3",
        [
            "PaddedInput_3",
            "alloc11",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc13",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_7",
        [
            "alloc",
            "tsid_100",
            "Conv2dOutput_7",
            "alloc1",
            "tsid_101",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc6",
        [
            "alloc5",
            "tsid_101",
            "tsid_100",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "alloc7",
            "PaddedInput_1",
            "alloc8",
            "Conv2dOutput_1",
            "PaddedInput_6",
            "alloc9",
            "Conv2dOutput_6",
            "PaddedInput_4",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_5",
        [
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "PaddedInput_5",
            "alloc5",
            "alloc11",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc7",
        [
            "Conv2dOutput_2",
            "tsid_100",
            "tsid_101",
            "alloc5",
            "alloc6",
            "PaddedInput_2",
            "PaddedInput_1",
            "alloc8",
            "Conv2dOutput_1",
            "PaddedInput_6",
            "alloc9",
            "Conv2dOutput_6",
            "PaddedInput_4",
            "alloc10",
            "Conv2dOutput_4",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "alloc11",
            "tensor_3",
            "alloc12",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "alloc13",
            "alloc14",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_2",
        [
            "alloc6",
            "alloc5",
            "tsid_101",
            "tsid_100",
            "Conv2dOutput_2",
            "alloc7",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_1",
        [
            "alloc7",
            "tsid_100",
            "tsid_101",
            "alloc5",
            "alloc6",
            "alloc8",
            "Conv2dOutput_1",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc11",
        [
            "Conv2dOutput_5",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "PaddedInput_5",
            "alloc5",
            "tensor_3",
            "alloc12",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "alloc13",
            "alloc14",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc",
        [
            "tsid_100",
            "PaddedInput_7",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc4",
        [
            "Conv2dOutput_8",
            "PaddedInput_8",
            "tsid_101",
            "tsid_100",
            "alloc5",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_6",
        [
            "alloc7",
            "tsid_100",
            "alloc8",
            "tsid_101",
            "alloc5",
            "alloc6",
            "alloc9",
            "Conv2dOutput_6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc2",
        [
            "tensor_2",
            "tsid_101",
            "tsid_100",
            "PaddedInput",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc12",
        [
            "tensor_3",
            "alloc11",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "PaddedInput_3",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_8",
        [
            "PaddedInput_8",
            "tsid_101",
            "tsid_100",
            "alloc4",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc10",
        [
            "PaddedInput_4",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc5",
            "Conv2dOutput_4",
            "PaddedInput_5",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tensor_2",
        [
            "alloc1",
            "tsid_101",
            "tsid_100",
            "alloc2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_2",
        [
            "PaddedInput_2",
            "alloc6",
            "alloc5",
            "tsid_101",
            "tsid_100",
            "alloc7",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_4",
        [
            "alloc10",
            "PaddedInput_4",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc5",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tsid_100",
        [
            "alloc",
            "PaddedInput_7",
            "Conv2dOutput_7",
            "alloc1",
            "tsid_101",
            "tensor_2",
            "alloc2",
            "PaddedInput",
            "Conv2dOutput",
            "alloc3",
            "PaddedInput_8",
            "Conv2dOutput_8",
            "alloc4",
            "alloc5",
            "alloc6",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "alloc7",
            "PaddedInput_1",
            "alloc8",
            "Conv2dOutput_1",
            "PaddedInput_6",
            "alloc9",
            "Conv2dOutput_6",
            "PaddedInput_4",
            "alloc10",
            "Conv2dOutput_4",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "alloc11",
            "tensor_3",
            "alloc12",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "alloc13",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc3",
        [
            "Conv2dOutput",
            "PaddedInput",
            "tsid_101",
            "tsid_100",
            "PaddedInput_8",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc5",
        [
            "alloc4",
            "tsid_101",
            "tsid_100",
            "alloc6",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "alloc7",
            "PaddedInput_1",
            "alloc8",
            "Conv2dOutput_1",
            "PaddedInput_6",
            "alloc9",
            "Conv2dOutput_6",
            "PaddedInput_4",
            "alloc10",
            "Conv2dOutput_4",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "alloc11",
            "tensor_3",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc14",
        [
            "alloc13",
            "alloc11",
            "alloc7",
            "alloc9",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput",
        [
            "alloc2",
            "tsid_101",
            "tsid_100",
            "Conv2dOutput",
            "alloc3",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_3",
        [
            "alloc12",
            "alloc11",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "Conv2dOutput_3",
            "alloc13",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_1",
        [
            "PaddedInput_1",
            "alloc7",
            "tsid_100",
            "alloc8",
            "tsid_101",
            "alloc5",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_6",
        [
            "PaddedInput_6",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc5",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_5",
        [
            "alloc10",
            "alloc7",
            "tsid_100",
            "alloc9",
            "tsid_101",
            "alloc5",
            "Conv2dOutput_5",
            "alloc11",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc13",
        [
            "PaddedInput_3",
            "alloc11",
            "alloc7",
            "tsid_100",
            "Conv2dOutput_3",
            "alloc9",
            "tsid_101",
            "alloc14",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_7",
        [
            "PaddedInput_7",
            "tsid_100",
            "alloc1",
            "tsid_101",
        ],
        buffer_info_map,
    )

    # check sizes
    assert buffer_info_map["Conv2dOutput"].size_bytes == 256
    assert buffer_info_map["PaddedInput_7"].size_bytes == 314646
    assert buffer_info_map["PaddedInput_6"].size_bytes == 172800
    assert buffer_info_map["alloc9"].size_bytes == 401408
    assert buffer_info_map["PaddedInput_2"].size_bytes == 301056
    assert buffer_info_map["alloc11"].size_bytes == 100352
    assert buffer_info_map["alloc6"].size_bytes == 1204224
    assert buffer_info_map["Conv2dOutput_4"].size_bytes == 4
    assert buffer_info_map["alloc2"].size_bytes == 1605632
    assert buffer_info_map["alloc13"].size_bytes == 100352
    assert buffer_info_map["alloc"].size_bytes == 1204224
    assert buffer_info_map["PaddedInput_8"].size_bytes == 430592
    assert buffer_info_map["tsid_100"].size_bytes == 1
    assert buffer_info_map["Conv2dOutput_2"].size_bytes == 256
    assert buffer_info_map["PaddedInput_3"].size_bytes == 301056
    assert buffer_info_map["tensor_3"].size_bytes == 150528
    assert buffer_info_map["Conv2dOutput_5"].size_bytes == 4
    assert buffer_info_map["Conv2dOutput_7"].size_bytes == 256
    assert buffer_info_map["PaddedInput_1"].size_bytes == 301056
    assert buffer_info_map["Conv2dOutput_6"].size_bytes == 256
    assert buffer_info_map["PaddedInput"].size_bytes == 401408
    assert buffer_info_map["alloc12"].size_bytes == 1204224
    assert buffer_info_map["alloc5"].size_bytes == 602112
    assert buffer_info_map["tensor_2"].size_bytes == 200704
    assert buffer_info_map["alloc10"].size_bytes == 100352
    assert buffer_info_map["alloc7"].size_bytes == 200704
    assert buffer_info_map["alloc3"].size_bytes == 1605632
    assert buffer_info_map["Conv2dOutput_8"].size_bytes == 256
    assert buffer_info_map["Conv2dOutput_3"].size_bytes == 4
    assert buffer_info_map["alloc8"].size_bytes == 602112
    assert buffer_info_map["tsid_101"].size_bytes == 1
    assert buffer_info_map["Conv2dOutput_1"].size_bytes == 4
    assert buffer_info_map["alloc4"].size_bytes == 2408448
    assert buffer_info_map["alloc1"].size_bytes == 3211264
    assert buffer_info_map["alloc14"].size_bytes == 100352
    assert buffer_info_map["PaddedInput_4"].size_bytes == 301056
    assert buffer_info_map["PaddedInput_5"].size_bytes == 28800


# fmt: off
@tvm.script.ir_module
class MultipleCallsToSamePrimFuncModule:
    @T.prim_func
    def tvmgen_default_fused_layout_transform_1(placeholder: T.handle, T_layout_trans: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_layout_transform_1", "tir.noalias": True})
        placeholder_1 = T.match_buffer(placeholder, [864], dtype="float32")
        T_layout_trans_1 = T.match_buffer(T_layout_trans, [41], dtype="float32")
        # body
        for ax0_ax1_fused_ax2_fused, ax3, ax4_inner in T.grid(24, 12, 3):
            T_layout_trans_1[ax0_ax1_fused_ax2_fused * 36 + ax3 * 3 + ax4_inner] = placeholder_1[ax4_inner * 288 + ax0_ax1_fused_ax2_fused * 12 + ax3]

    @T.prim_func
    def tvmgen_default_fused_nn_contrib_conv2d_NCHWc(placeholder_2: T.handle, placeholder_3: T.handle, conv2d_NCHWc: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_nn_contrib_conv2d_NCHWc", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [864], dtype="float32")
        placeholder_5 = T.match_buffer(placeholder_3, [81], dtype="float32")
        conv2d_NCHWc_1 = T.match_buffer(conv2d_NCHWc, [41], dtype="float32")
        # body
        data_pad = T.allocate([1092], "float32", "global")
        for i0_i1_fused_i2_fused, i3, i4 in T.grid(26, 14, 3):
            data_pad[i0_i1_fused_i2_fused * 42 + i3 * 3 + i4] = T.if_then_else(1 <= i0_i1_fused_i2_fused and i0_i1_fused_i2_fused < 25 and 1 <= i3 and i3 < 13, placeholder_4[i0_i1_fused_i2_fused * 36 + i3 * 3 + i4 - 39], T.float32(0), dtype="float32")
        for n_oc_chunk_fused_oh_fused in T.serial(0, 24):
            conv2d_NCHWc_global = T.allocate([36], "float32", "global")
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 3] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 6] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 9] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 12] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 15] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 18] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 21] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 24] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 27] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 30] = T.float32(0)
            for oc_block_c_init in T.serial(0, 3):
                conv2d_NCHWc_global[oc_block_c_init + 33] = T.float32(0)
            for kh, kw, ic_inner in T.grid(3, 3, 3):
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c] = conv2d_NCHWc_global[oc_block_c] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 3] = conv2d_NCHWc_global[oc_block_c + 3] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 3] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 6] = conv2d_NCHWc_global[oc_block_c + 6] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 6] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 9] = conv2d_NCHWc_global[oc_block_c + 9] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 9] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 12] = conv2d_NCHWc_global[oc_block_c + 12] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 12] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 15] = conv2d_NCHWc_global[oc_block_c + 15] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 15] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 18] = conv2d_NCHWc_global[oc_block_c + 18] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 18] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 21] = conv2d_NCHWc_global[oc_block_c + 21] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 21] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 24] = conv2d_NCHWc_global[oc_block_c + 24] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 24] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 27] = conv2d_NCHWc_global[oc_block_c + 27] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 27] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 30] = conv2d_NCHWc_global[oc_block_c + 30] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 30] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
                for oc_block_c in T.serial(0, 3):
                    conv2d_NCHWc_global[oc_block_c + 33] = conv2d_NCHWc_global[oc_block_c + 33] + data_pad[kh * 42 + n_oc_chunk_fused_oh_fused * 42 + kw * 3 + ic_inner + 33] * placeholder_5[kh * 27 + kw * 9 + ic_inner * 3 + oc_block_c]
            for ow_inner, oc_block in T.grid(12, 3):
                conv2d_NCHWc_1[n_oc_chunk_fused_oh_fused * 36 + ow_inner * 3 + oc_block] = conv2d_NCHWc_global[ow_inner * 3 + oc_block]

    @T.prim_func
    def tvmgen_default_fused_nn_softmax_add_add_multiply_add(placeholder_6: T.handle, placeholder_7: T.handle, placeholder_8: T.handle, placeholder_9: T.handle, placeholder_10: T.handle, T_add: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_nn_softmax_add_add_multiply_add", "tir.noalias": True})
        placeholder_11 = T.match_buffer(placeholder_6, [864], dtype="float32")
        placeholder_12 = T.match_buffer(placeholder_7, [864], dtype="float32")
        placeholder_13 = T.match_buffer(placeholder_8, [3], dtype="float32")
        placeholder_14 = T.match_buffer(placeholder_9, [3], dtype="float32")
        placeholder_15 = T.match_buffer(placeholder_10, [3], dtype="float32")
        T_add_1 = T.match_buffer(T_add, [864], dtype="float32")
        # body
        for ax0_ax1_fused_ax2_fused in T.serial(0, 72):
            T_softmax_norm = T.allocate([12], "float32", "global")
            with T.allocate([1], "float32", "global") as T_softmax_maxelem:
                T_softmax_maxelem[0] = T.float32(-3.4028234663852886e+38)
                for k in T.serial(0, 12):
                    T_softmax_maxelem[0] = T.max(T_softmax_maxelem[0], placeholder_11[ax0_ax1_fused_ax2_fused * 12 + k])
                T_softmax_exp = T.allocate([12], "float32", "global")
                for i3 in T.serial(0, 12):
                    T_softmax_exp[i3] = T.exp(placeholder_11[ax0_ax1_fused_ax2_fused * 12 + i3] - T_softmax_maxelem[0], dtype="float32")
                T_softmax_expsum = T.allocate([1], "float32", "global")
                T_softmax_expsum[0] = T.float32(0)
                for k in T.serial(0, 12):
                    T_softmax_expsum[0] = T_softmax_expsum[0] + T_softmax_exp[k]
                for i3 in T.serial(0, 12):
                    T_softmax_norm[i3] = T_softmax_exp[i3] / T_softmax_expsum[0]
            for ax3 in T.serial(0, 12):
                T_add_1[ax0_ax1_fused_ax2_fused * 12 + ax3] = (placeholder_12[ax0_ax1_fused_ax2_fused * 12 + ax3] + T_softmax_norm[ax3] + placeholder_13[T.floordiv(ax0_ax1_fused_ax2_fused, 24)]) * placeholder_14[T.floordiv(ax0_ax1_fused_ax2_fused, 24)] + placeholder_15[T.floordiv(ax0_ax1_fused_ax2_fused, 24)]

    @T.prim_func
    def tvmgen_default_fused_nn_contrib_dense_pack_nn_relu(placeholder_16: T.handle, placeholder_17: T.handle, T_relu: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_nn_contrib_dense_pack_nn_relu", "tir.noalias": True})
        placeholder_18 = T.match_buffer(placeholder_16, [864], dtype="float32")
        placeholder_19 = T.match_buffer(placeholder_17, [144], dtype="float32")
        T_relu_1 = T.match_buffer(T_relu, [864], dtype="float32")
        # body
        for ax1_outer_ax0_outer_fused in T.serial(0, 18):
            compute = T.allocate([48], "float32", "global")
            with T.allocate([48], "float32", "global") as compute_global:
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 6] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 12] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 18] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 24] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 30] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 36] = T.float32(0)
                for x_c_init in T.serial(0, 6):
                    compute_global[x_c_init + 42] = T.float32(0)
                for k_outer in T.serial(0, 12):
                    for x_c in T.serial(0, 6):
                        compute_global[x_c] = compute_global[x_c] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 6] = compute_global[x_c + 6] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 12] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 12] = compute_global[x_c + 12] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 24] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 18] = compute_global[x_c + 18] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 36] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 24] = compute_global[x_c + 24] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 48] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 30] = compute_global[x_c + 30] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 60] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 36] = compute_global[x_c + 36] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 72] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                    for x_c in T.serial(0, 6):
                        compute_global[x_c + 42] = compute_global[x_c + 42] + placeholder_18[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + k_outer + 84] * placeholder_19[T.floordiv(ax1_outer_ax0_outer_fused, 9) * 72 + k_outer * 6 + x_c]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner] = compute_global[x_inner_inner]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 6] = compute_global[x_inner_inner + 6]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 12] = compute_global[x_inner_inner + 12]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 18] = compute_global[x_inner_inner + 18]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 24] = compute_global[x_inner_inner + 24]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 30] = compute_global[x_inner_inner + 30]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 36] = compute_global[x_inner_inner + 36]
                for x_inner_inner in T.serial(0, 6):
                    compute[x_inner_inner + 42] = compute_global[x_inner_inner + 42]
            for ax0_inner_inner, ax1_inner_inner in T.grid(8, 6):
                T_relu_1[T.floormod(ax1_outer_ax0_outer_fused, 9) * 96 + ax0_inner_inner * 12 + T.floordiv(ax1_outer_ax0_outer_fused, 9) * 6 + ax1_inner_inner] = T.max(compute[ax0_inner_inner * 6 + ax1_inner_inner], T.float32(0))

    @T.prim_func
    def tvmgen_default_fused_reshape_1(placeholder_20: T.handle, T_reshape: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_reshape_1", "tir.noalias": True})
        placeholder_21 = T.match_buffer(placeholder_20, [864], dtype="float32")
        T_reshape_1 = T.match_buffer(T_reshape, [864], dtype="float32")
        # body
        for ax0, ax1_inner in T.grid(72, 12):
            T_reshape_1[ax0 * 12 + ax1_inner] = placeholder_21[ax0 * 12 + ax1_inner]

    @T.prim_func
    def tvmgen_default_fused_layout_transform(placeholder_22: T.handle, T_layout_trans_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_layout_transform", "tir.noalias": True})
        placeholder_23 = T.match_buffer(placeholder_22, [864], dtype="float32")
        T_layout_trans_3 = T.match_buffer(T_layout_trans_2, [864], dtype="float32")
        # body
        for ax0_ax1_fused, ax2, ax3_inner in T.grid(3, 24, 12):
            T_layout_trans_3[ax0_ax1_fused * 288 + ax2 * 12 + ax3_inner] = placeholder_23[ax2 * 36 + ax3_inner * 3 + ax0_ax1_fused]

    @T.prim_func
    def tvmgen_default_fused_reshape(placeholder_24: T.handle, T_reshape_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_reshape", "tir.noalias": True})
        placeholder_25 = T.match_buffer(placeholder_24, [864], dtype="float32")
        T_reshape_3 = T.match_buffer(T_reshape_2, [864], dtype="float32")
        # body
        for ax0_ax1_fused, ax2, ax3_inner in T.grid(3, 24, 12):
            T_reshape_3[ax0_ax1_fused * 288 + ax2 * 12 + ax3_inner] = placeholder_25[ax0_ax1_fused * 288 + ax2 * 12 + ax3_inner]

    @T.prim_func
    def tvmgen_default_fused_nn_softmax_add(placeholder_26: T.handle, placeholder_27: T.handle, T_add_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "tvmgen_default_fused_nn_softmax_add", "tir.noalias": True})
        placeholder_28 = T.match_buffer(placeholder_26, [864], dtype="float32")
        placeholder_29 = T.match_buffer(placeholder_27, [864], dtype="float32")
        T_add_3 = T.match_buffer(T_add_2, [864], dtype="float32")
        # body
        for ax0_ax1_fused_ax2_fused in T.serial(0, 72):
            T_softmax_norm = T.allocate([12], "float32", "global")
            with T.allocate([1], "float32", "global") as T_softmax_maxelem:
                T_softmax_maxelem[0] = T.float32(-3.4028234663852886e+38)
                for k in T.serial(0, 12):
                    T_softmax_maxelem[0] = T.max(T_softmax_maxelem[0], placeholder_28[ax0_ax1_fused_ax2_fused * 12 + k])
                T_softmax_exp = T.allocate([12], "float32", "global")
                for i3 in T.serial(0, 12):
                    T_softmax_exp[i3] = T.exp(placeholder_28[ax0_ax1_fused_ax2_fused * 12 + i3] - T_softmax_maxelem[0], dtype="float32")
                T_softmax_expsum = T.allocate([1], "float32", "global")
                T_softmax_expsum[0] = T.float32(0)
                for k in T.serial(0, 12):
                    T_softmax_expsum[0] = T_softmax_expsum[0] + T_softmax_exp[k]
                for i3 in T.serial(0, 12):
                    T_softmax_norm[i3] = T_softmax_exp[i3] / T_softmax_expsum[0]
            for ax3 in T.serial(0, 12):
                T_add_3[ax0_ax1_fused_ax2_fused * 12 + ax3] = placeholder_29[ax0_ax1_fused_ax2_fused * 12 + ax3] + T_softmax_norm[ax3]

    @R.function
    def run_model(input: Tensor((16, 16), "uint8")) -> Tensor:
        tsid_100 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_101 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        tsid_102 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        sid_8 = relax.builtin.alloc_tensor((3456, 1), runtime_device_index=0, dtype="int8")

        sid_23 = relax.call_tir("tvmgen_default_fused_layout_transform_1", (input), (2, 1), dtype="int32")
        sid_7 = relax.call_tir("tvmgen_default_fused_nn_contrib_conv2d_NCHWc", (sid_8, tsid_100), (3456, 1), dtype="int32")
        sid_6 = relax.call_tir("tvmgen_default_fused_layout_transform", (sid_7), (3456, 1), dtype="int32")
        sid_12 = relax.call_tir("tvmgen_default_fused_reshape_1", (input), (3456, 1), dtype="int32")
        sid_11 = relax.call_tir("tvmgen_default_fused_nn_contrib_dense_pack_nn_relu", (sid_12, tsid_100), (3456, 1), dtype="int32")
        sid_10 = relax.call_tir("tvmgen_default_fused_reshape", (sid_11), (3456, 1), dtype="int32")
        sid_5 = relax.call_tir("tvmgen_default_fused_nn_softmax_add_add_multiply_add", (sid_6, sid_10, tsid_100, tsid_101, tsid_102), (3456, 1), dtype="int32")
        sid_4 = relax.call_tir("tvmgen_default_fused_layout_transform_1", (sid_5), (3456, 1), dtype="int32")
        sid_3 = relax.call_tir("tvmgen_default_fused_nn_contrib_conv2d_NCHWc", (sid_4, tsid_100), (3456, 1), dtype="int32")
        sid_2 = relax.call_tir("tvmgen_default_fused_layout_transform", (sid_3), (3456, 1), dtype="int32")
        sid_20 = relax.call_tir("tvmgen_default_fused_reshape_1", (sid_5), (3456, 1), dtype="int32")
        sid_19 = relax.call_tir("tvmgen_default_fused_nn_contrib_dense_pack_nn_relu", (sid_20, tsid_100), (3456, 1), dtype="int32")
        sid_18 = relax.call_tir("tvmgen_default_fused_reshape", (sid_19), (3456, 1), dtype="int32")
        output = relax.call_tir("tvmgen_default_fused_nn_softmax_add", (sid_2, sid_18), (3456, 1), dtype="int32")
        return output
# fmt: on


def test_multiple_calls_to_same_primfunc():
    target = Target("c")
    global_ws_pool = WorkspacePoolInfo(
        pool_name="global_workspace",
        targets=[target],
    )
    global_const_pool = ConstantPoolInfo(
        pool_name="global_constants",
        targets=[target],
    )

    relax_mod = MultipleCallsToSamePrimFuncModule
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)
    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)
    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(
        relax_mod, [global_ws_pool], [global_const_pool]
    )
    main_func = relax_mod["run_model"]
    buffer_info_analysis = tvm.relax.analysis.extract_buffer_info(main_func, relax_mod)
    assert buffer_info_analysis.memory_pressure == 41857
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_analysis.buffer_info_stmts)

    # check conflicts
    _verify_conflicts(
        "T_softmax_norm",
        [
            "alloc5",
            "tsid_100",
            "alloc2",
            "T_softmax_maxelem",
            "T_softmax_expsum",
            "T_softmax_exp",
            "tsid_101",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc10",
        [
            "alloc9",
            "alloc6",
            "tsid_100",
            "compute",
            "compute_global",
            "alloc11",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc11",
        [
            "compute_global",
            "compute",
            "alloc9",
            "alloc10",
            "tsid_100",
            "alloc12",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "compute",
        [
            "alloc4",
            "compute_global",
            "alloc3",
            "alloc2",
            "tsid_100",
            "alloc9",
            "alloc10",
            "tsid_100",
            "compute_global",
            "alloc11",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc1",
        [
            "tsid_100",
            "data_pad",
            "conv2d_NCHWc_global",
            "alloc2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc2",
        [
            "alloc1",
            "tsid_100",
            "alloc3",
            "alloc4",
            "compute_global",
            "compute",
            "alloc5",
            "T_softmax_norm",
            "T_softmax_maxelem",
            "T_softmax_expsum",
            "T_softmax_exp",
            "tsid_101",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc8",
        [
            "conv2d_NCHWc_global",
            "data_pad",
            "alloc6",
            "tsid_100",
            "alloc9",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc12",
        [
            "alloc9",
            "alloc11",
            "T_softmax_expsum2",
            "alloc13",
            "T_softmax_norm2",
            "T_softmax_exp2",
            "T_softmax_maxelem2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tsid_100",
        [
            "conv2d_NCHWc_global",
            "data_pad",
            "alloc1",
            "alloc2",
            "alloc3",
            "alloc4",
            "compute_global",
            "compute",
            "alloc5",
            "T_softmax_norm",
            "T_softmax_maxelem",
            "T_softmax_expsum",
            "T_softmax_exp",
            "tsid_101",
            "tsid_102",
            "alloc6",
            "alloc7",
            "data_pad",
            "conv2d_NCHWc_global",
            "alloc8",
            "alloc9",
            "alloc10",
            "compute",
            "compute_global",
            "alloc11",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_maxelem2",
        [
            "T_softmax_exp2",
            "T_softmax_norm2",
            "alloc13",
            "T_softmax_expsum2",
            "alloc12",
            "alloc9",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_maxelem",
        [
            "T_softmax_norm",
            "alloc5",
            "tsid_100",
            "alloc2",
            "T_softmax_expsum",
            "T_softmax_exp",
            "tsid_101",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_norm2",
        [
            "alloc13",
            "T_softmax_expsum2",
            "alloc12",
            "alloc9",
            "T_softmax_exp2",
            "T_softmax_maxelem2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_exp2",
        [
            "T_softmax_norm2",
            "alloc13",
            "T_softmax_expsum2",
            "alloc12",
            "alloc9",
            "T_softmax_maxelem2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_expsum2",
        [
            "alloc12",
            "alloc9",
            "alloc13",
            "T_softmax_norm2",
            "T_softmax_exp2",
            "T_softmax_maxelem2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc5",
        [
            "tsid_100",
            "alloc2",
            "alloc4",
            "T_softmax_norm",
            "T_softmax_maxelem",
            "T_softmax_expsum",
            "T_softmax_exp",
            "tsid_101",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc",
        [],
        buffer_info_map,
    )
    _verify_conflicts(
        "sid_8",
        [
            "data_pad",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc4",
        [
            "alloc3",
            "alloc2",
            "tsid_100",
            "compute_global",
            "compute",
            "alloc5",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "compute_global",
        [
            "alloc4",
            "alloc3",
            "alloc2",
            "tsid_100",
            "compute",
            "compute",
            "alloc9",
            "alloc10",
            "tsid_100",
            "alloc11",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "data_pad",
        [
            "sid_8",
            "conv2d_NCHWc_global",
            "tsid_100",
            "alloc1",
            "alloc7",
            "alloc6",
            "tsid_100",
            "conv2d_NCHWc_global",
            "alloc8",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc3",
        [
            "alloc2",
            "tsid_100",
            "alloc4",
            "compute_global",
            "compute",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_expsum",
        [
            "T_softmax_maxelem",
            "T_softmax_norm",
            "alloc5",
            "tsid_100",
            "alloc2",
            "T_softmax_exp",
            "tsid_101",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "T_softmax_exp",
        [
            "T_softmax_maxelem",
            "T_softmax_norm",
            "T_softmax_expsum",
            "alloc5",
            "tsid_100",
            "alloc2",
            "tsid_101",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc7",
        [
            "alloc6",
            "tsid_100",
            "data_pad",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tsid_101",
        [
            "T_softmax_exp",
            "T_softmax_maxelem",
            "T_softmax_norm",
            "T_softmax_expsum",
            "alloc5",
            "tsid_100",
            "alloc2",
            "tsid_102",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc9",
        [
            "alloc6",
            "alloc8",
            "tsid_100",
            "alloc10",
            "compute",
            "compute_global",
            "alloc11",
            "alloc12",
            "T_softmax_expsum2",
            "alloc13",
            "T_softmax_norm2",
            "T_softmax_exp2",
            "T_softmax_maxelem2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "conv2d_NCHWc_global",
        [
            "data_pad",
            "tsid_100",
            "alloc1",
            "data_pad",
            "alloc6",
            "tsid_100",
            "alloc8",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "tsid_102",
        [
            "tsid_101",
            "T_softmax_exp",
            "T_softmax_maxelem",
            "T_softmax_norm",
            "T_softmax_expsum",
            "alloc5",
            "tsid_100",
            "alloc2",
            "alloc6",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc13",
        [
            "T_softmax_expsum2",
            "alloc12",
            "alloc9",
            "T_softmax_norm2",
            "T_softmax_exp2",
            "T_softmax_maxelem2",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "alloc6",
        [
            "tsid_102",
            "tsid_101",
            "T_softmax_exp",
            "T_softmax_maxelem",
            "T_softmax_norm",
            "T_softmax_expsum",
            "alloc5",
            "tsid_100",
            "alloc2",
            "alloc7",
            "data_pad",
            "conv2d_NCHWc_global",
            "alloc8",
            "alloc9",
            "alloc10",
        ],
        buffer_info_map,
    )


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
