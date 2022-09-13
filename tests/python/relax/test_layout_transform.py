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

import tvm
import tvm.testing
from tvm import relax, relay
from tvm.relax.testing import relay_translator
from tvm import meta_schedule as ms
from tvm.target import Target
import numpy as np
import pytest
import tempfile
from tvm.script import tir as T
from tvm.relay import analysis, transform


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def print_relay(before, after=None, expected=None):
    print("\n** F0: RELAY BEFORE LAYOUT CONVERSION **\n")
    print(before)
    if after:
        print("\n** F1: RELAY BEFORE LAYOUT CONVERSION **\n")
        print(after)
    if expected:
        print("\n** EXPECTED **\n")
        print(expected)


def print_relax(before, after=None, expected=None):
    print(
        "\n** F2: (F0 --> RelayRelaxTranslator --> F2) RELAX FROM RELAY WITHOUT RELAY LAYOUT CONVERSION **\n"
    )
    before.show()
    if after:
        print(
            "\n** F3: (F1 --> RelayRelaxTranslator --> F3) RELAX FROM RELAY WITH RELAY LAYOUT CONVERSION **\n"
        )
        after.show()
    if expected:
        print("\n** EXPECTED **\n")
        expected.show()


def print_flow(func):
    print("\n** F4: (F2 --> Flow Constraints in Relax --> F4) RELAX AFTER FLOWING CONSTRAINTS **\n")
    func.show()


def apply_conv_layout_conversion(relay_before, new_layouts=["NCHW", "OIHW"]):
    relay_after = run_opt_pass(relay_before, transform.ConvertLayout({"nn.conv2d": new_layouts}))
    print_relay(relay_before, relay_after)
    relax_before = relay_translator.from_relay(relay_before, "llvm")
    relax_after = relay_translator.from_relay(relay_after, "llvm")

    return relay_before, relay_after, relax_before, relax_after


def flow_constraint(func: relax.Function, block_name: str, read_index_map, write_index_map=None):
    if not write_index_map:
        write_index_map = read_index_map
    sch = tvm.tir.Schedule(func)
    sch.transform_layout(
        block_name,
        ("read", 0),
        index_map=read_index_map,
    )
    sch.transform_layout(
        block_name,
        ("write", 0),
        index_map=write_index_map,
    )
    return sch.mod["main"]


def test_reduce():
    reduce_op = relay.sum

    def before():
        x = relay.var("x", shape=(2, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = reduce_op(y, axis=[1, 2])
        y = relay.Function([x, weight], y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["sum"], relax_after["sum"])

    sum_after = flow_constraint(
        relax_before["sum"],
        "rxplaceholder_red",
        lambda n, h, w, c: (n, c // 4, h, w, c % 4),
        write_index_map=lambda n, c: (n, c // 4, c % 4),
    )
    print_flow(sum_after)


def test_binary_broadcast():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        bias = relay.var("bias", shape=(32, 56, 56, 1))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = y + bias
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["add"], relax_after["add"])

    # create the add primfunc to get around the int64 vs int32 issue created when converitng
    # relay to relax.
    @T.prim_func
    def add(
        rxplaceholder: T.Buffer[(32, 56, 56, 64), "float32"],
        rxplaceholder_1: T.Buffer[(32, 56, 56, 1), "float32"],
        T_add: T.Buffer[(32, 56, 56, 64), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2, i3 in T.grid(32, 56, 56, 64):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(rxplaceholder[ax0, ax1, ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, 0])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = (
                    rxplaceholder[ax0, ax1, ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, 0]
                )

    sch = tvm.tir.Schedule(add)
    add_after = flow_constraint(add, "T_add", lambda n, h, w, c: (n, c // 4, h, w, c % 4))
    print_flow(add_after)


def test_strided_slice():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.strided_slice(y, begin=[0, 0, 0, 1], end=[1, 10, 56, -1], strides=[1, 2, 1, 1])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())

    print_relax(relax_before["strided_slice"], relax_after["strided_slice"])

    strided_slice_after = flow_constraint(
        relax_before["strided_slice"],
        "T_strided_slice",
        lambda n, h, w, c: (n, c, h, w),
    )
    print_flow(strided_slice_after)


def test_pool():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["max_pool2d"], relax_after["max_pool2d"])

    pool_after = flow_constraint(
        relax_before["max_pool2d"], "tensor", lambda n, h, w, c: (n, c // 4, h, w, c % 4)
    )
    print_flow(pool_after)


def test_elemwise():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["relu"], relax_after["relu"])

    relu_after = flow_constraint(
        relax_before["relu"], "T_relu", lambda n, h, w, c: (n, c // 4, h, w, c % 4)
    )
    print_flow(relu_after)


def test_conv():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["conv2d"], relax_after["contrib_conv2d_NCHWc"])

    @T.prim_func
    def conv2d(
        rxplaceholder: T.Buffer[(32, 56, 56, 64), "float32"],
        rxplaceholder_1: T.Buffer[(3, 3, 64, 64), "float32"],
        conv2d_nhwc: T.Buffer[(32, 56, 56, 64), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "conv2d", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([32, 58, 58, 64], dtype="float32")
        for i0, i1, i2, i3 in T.grid(32, 58, 58, 64):
            with T.block("pad_temp"):
                i0_1 = T.axis.spatial(32, i0)
                i1_1 = T.axis.spatial(58, i1)
                i2_1 = T.axis.spatial(58, i2)
                i3_1 = T.axis.spatial(64, i3)
                T.reads(rxplaceholder[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                    1 <= i1_1 and i1_1 < 57 and 1 <= i2_1 and i2_1 < 57,
                    rxplaceholder[i0_1, i1_1 - 1, i2_1 - 1, i3_1],
                    T.float32(0),
                    dtype="float32",
                )
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(32, 56, 56, 64, 3, 3, 64):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], rxplaceholder_1[ry, rx, rc, ff])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = T.float32(0)
                conv2d_nhwc[nn, yy, xx, ff] = (
                    conv2d_nhwc[nn, yy, xx, ff]
                    + pad_temp[nn, yy + ry, xx + rx, rc] * rxplaceholder_1[ry, rx, rc, ff]
                )

    conv2d_after = flow_constraint(
        conv2d, "conv2d_nhwc", lambda n, h, w, c: (n, c // 4, h, w, c % 4)
    )
    print_flow(conv2d_after)


def test_upsampling():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.upsampling(y, scale_h=2, scale_w=3)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(
        before(), new_layouts=["NCHW16c", "OIHW16i16o"]
    )
    print_relax(relax_before["upsampling"], relax_after["upsampling"])

    upsampling_after = flow_constraint(
        relax_before["upsampling"], "resize", lambda n, c, h, w, cb: (n, c // 4, h, w, c % 4, cb)
    )
    print_flow(upsampling_after)
    upsampling_after = flow_constraint(
        upsampling_after, "resize", lambda n, c, h, w, cb1, cb0: (n, c, h, w, cb1 * 4 + cb0)
    )
    print_flow(upsampling_after)


def test_transpose():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.transpose(y, axes=[0, 3, 1, 2])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())

    print_relax(relax_before["transpose"], relax_after["transpose"])

    transpose_after = flow_constraint(
        relax_before["transpose"],
        "T_transpose",
        lambda n, h, w, c: (n, c, h, w),
        write_index_map=lambda n, c, h, w: (n, c, h, w),
    )
    print_flow(transpose_after)


def test_pad():
    def before():
        x = relay.var("x", shape=(32, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.pad(y, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())

    print_relax(relax_before["pad"], relax_after["pad"])

    pad_after = flow_constraint(relax_before["pad"], "T_pad", lambda n, h, w, c: (n, c, h, w))
    print_flow(pad_after)


if __name__ == "__main__":
    pytest.main([__file__])
