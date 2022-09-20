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
        print("\n** F1: RELAY AFTER LAYOUT CONVERSION **\n")
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


def apply_conv_layout_conversion(relay_before, new_layouts=["NHWC", "HWIO"]):
    relay_after = run_opt_pass(relay_before, transform.ConvertLayout({"nn.conv2d": new_layouts}))
    print_relay(relay_before, relay_after)
    relax_before = relay_translator.from_relay(
        relay_before, "llvm", disabled_pass=["AlterOpLayout"]
    )
    relax_after = relay_translator.from_relay(relay_after, "llvm", disabled_pass=["AlterOpLayout"])

    return relay_before, relay_after, relax_before, relax_after


def flow_constraint(
    func: relax.Function,
    block_name: str,
    read_indices=[0],
    write_indices=[0],
    read_index_map=lambda N, C, H, W: (N, H, W, C),
    write_index_map=None,
):
    if not write_index_map:
        write_index_map = read_index_map
    sch = tvm.tir.Schedule(func)
    for read_idx in read_indices:
        sch.transform_layout(
            block_name,
            ("read", read_idx),
            index_map=read_index_map,
        )
    for write_idx in write_indices:
        sch.transform_layout(
            block_name,
            ("write", write_idx),
            index_map=write_index_map,
        )
    return sch.mod["main"]


def test_elemwise():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["relu"], relax_after["relu"])

    relu_after = flow_constraint(
        relax_before["relu"],
        "T_relu",
        read_index_map=lambda N, C, H, W: (N, H, W, C),
        write_index_map=lambda i0, i1, i2, i3: (i0, i1, i2, i3),
    )
    print_flow(relu_after)


def test_pool():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["max_pool2d"], relax_after["max_pool2d"])

    pool_after = flow_constraint(relax_before["max_pool2d"], "tensor")
    print_flow(pool_after)


def test_conv():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["conv2d"], relax_after["conv2d"])


def test_reduce():
    reduce_op = relay.sum

    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = reduce_op(y, axis=[2, 3])
        y = relay.Function([x, weight], y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["sum"], relax_after["sum"])

    sum_after = flow_constraint(
        relax_before["sum"],
        "rxplaceholder_red",
        read_index_map=lambda N, C, H, W: (N, H, W, C),
        write_index_map=lambda N, C: (N, C),
    )
    print_flow(sum_after)


def test_upsampling():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.upsampling(y, scale_h=2, scale_w=3)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["upsampling"], relax_after["upsampling"])
    print_flow(relax_before["upsampling"])


def test_strided_slice():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.strided_slice(y, begin=[0, 0, 0, 1], end=[1, 10, 56, -1], strides=[1, 2, 1, 1])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())

    print_relax(relax_before["strided_slice"], relax_after["strided_slice"])

    strided_slice_after = flow_constraint(relax_before["strided_slice"], "T_strided_slice")
    print_flow(strided_slice_after)


def test_binary_broadcast():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        bias = relay.var("bias", shape=(32, 1, 56, 56))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = y + bias
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["add"], relax_after["add"])

    # Relay to Relax conversion changes the tensor dimensions from int32 to int64. This
    # causes some int32 vs int64 bug when flowing constraints. As a workaround we create
    # an identical add PrimFunc with int32 dimensions.
    @T.prim_func
    def add(
        rxplaceholder: T.Buffer[(32, 64, 56, 56), "float32"],
        rxplaceholder_1: T.Buffer[(32, 1, 56, 56), "float32"],
        T_add: T.Buffer[(32, 64, 56, 56), "float32"],
    ) -> None:
        for i0, i1, i2, i3 in T.grid(32, 64, 56, 56):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(rxplaceholder[ax0, ax1, ax2, ax3], rxplaceholder_1[ax0, 0, ax2, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = (
                    rxplaceholder[ax0, ax1, ax2, ax3] + rxplaceholder_1[ax0, 0, ax2, ax3]
                )

    add_after = flow_constraint(add, "T_add")
    print_flow(add_after)


def test_transpose():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.transpose(y, axes=[0, 3, 1, 2])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())

    print_relax(relax_before["transpose"], relax_after["transpose"])

    transpose_after = flow_constraint(
        relax_before["transpose"],
        "T_transpose",
        read_index_map=lambda N, C, H, W: (N, H, W, C),
        write_index_map=lambda i0, i1, i2, i3: (i0, i1, i2, i3),
    )
    print_flow(transpose_after)


def test_pad():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.pad(y, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["pad"], relax_after["pad"])
    pad_after = flow_constraint(relax_before["pad"], "T_pad")
    print_flow(pad_after)


def test_split():
    def before():
        x = relay.var("x", shape=(32, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.split(y, indices_or_sections=2, axis=2).astuple()
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(before())
    print_relax(relax_before["split"], relax_after["split"])
    # apply constraints to input and first output
    split_after = flow_constraint(relax_before["split"], "T_split_sections")
    # apply constraints to second output
    split_after = flow_constraint(split_after, "T_split_sections_1", read_indices=[])
    print_flow(split_after)


if __name__ == "__main__":
    pytest.main([__file__])
