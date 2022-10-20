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

import tvm
from tvm import topi
import tvm.testing
from tvm import relax, relay
from tvm.relax.testing import relay_translator
from tvm import meta_schedule as ms
from tvm.target import Target
import numpy as np
import pytest
import tempfile
from tvm.script import tir as T, relax as R
from tvm.relay import analysis, transform
from tvm.runtime import vm as vm_rt


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


def apply_conv_layout_conversion(relay_before, new_layouts=["NHWC", "default"]):
    relay_after = run_opt_pass(relay_before, transform.ConvertLayout({"nn.conv2d": new_layouts}))
    print_relay(relay_before, relay_after)
    relax_before = relay_translator.from_relay(relay_before, "llvm")
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


data_transform = lambda N, H, W, C: (N, H // 8, W // 8, C // 32, H % 8, W % 8, C % 32)
filter_transform = lambda FH, FW, C, K: (K // 32, C // 32, FH, FW, (C % 32) // 4, K % 32, C % 4)


def relax_graph(dtype="float32", add_constraints=False):
    N, C, H, W = 32, 3, 224, 224
    K1, FH1, FW1 = 64, 5, 5
    K2, FH2, FW2 = 256, 7, 7
    data_layout = "NCHW"
    kernel_layout = "OIHW"
    if add_constraints:
        data_layout = "NHWC"
        kernel_layout = "HWIO"

    bb = relax.BlockBuilder()
    x = relax.Var("x", [N, C, H, W], relax.DynTensorType(4, dtype))
    f1 = relax.Var("f1", [K1, C, FH1, FW1], relax.DynTensorType(4, dtype))
    f2 = relax.Var("f2", [K2, C, FH2, FW2], relax.DynTensorType(4, dtype))
    bias = relax.const(np.random.randn(K2, 1, 1), dtype)
    with bb.function("main", [x, f1, f2]):
        with bb.dataflow():
            relu = bb.emit_te(topi.nn.relu, x)
            if add_constraints:
                relu = bb.emit_te(topi.layout_transform, relu, "NCHW", data_layout)
                f1 = bb.emit_te(topi.layout_transform, f1, "OIHW", kernel_layout)
            first_conv = bb.emit_te(
                topi.nn.conv2d,
                relu,
                f1,
                strides=1,
                padding=0,
                dilation=1,
                data_layout=data_layout,
                kernel_layout=kernel_layout,
                primfunc_name_hint="first_conv",
            )
            if add_constraints:
                first_conv = bb.emit_te(topi.layout_transform, first_conv, data_layout, "NCHW")
            pool = bb.emit_te(
                topi.nn.pool2d,
                first_conv,
                [2, 2],
                [1, 1],
                [1, 1],
                [0, 0, 0, 0],
                "max",
            )
            if add_constraints:
                pool = bb.emit_te(topi.layout_transform, pool, "NCHW", data_layout)
                f2 = bb.emit_te(topi.layout_transform, f2, "OIHW", kernel_layout)
            second_conv = bb.emit_te(
                topi.nn.conv2d,
                pool,
                f2,
                strides=1,
                padding=0,
                dilation=1,
                data_layout=data_layout,
                kernel_layout=kernel_layout,
                primfunc_name_hint="second_conv",
            )
            if add_constraints:
                second_conv = bb.emit_te(topi.layout_transform, second_conv, data_layout, "NCHW")

            add = bb.emit_te(topi.add, second_conv, bias)
            reduce = bb.emit_te(topi.sum, add, (2, 3), primfunc_name_hint="reduce")
            gv = bb.emit_output(reduce)
        bb.emit_func_output(gv)

    return bb.get()


def relax_graph_tiny(dtype="float32", add_constraints=False):
    N, C, H, W = 32, 64, 224, 224
    K1, FH1, FW1 = 64, 5, 5
    data_layout = "NCHW"
    kernel_layout = "OIHW"
    if add_constraints:
        data_layout = "NCHW4c"
        kernel_layout = "OIHW4i4o"

    bb = relax.BlockBuilder()
    x = relax.Var("x", [N, C, H, W], relax.DynTensorType(4, dtype))
    f1 = relax.Var("f1", [K1, C, FH1, FW1], relax.DynTensorType(4, dtype))
    bias = relax.const(np.random.randn(K1, 1, 1), dtype)
    with bb.function("main", [x, f1]):
        with bb.dataflow():
            if add_constraints:
                x = bb.emit_te(topi.layout_transform, x, "NCHW", data_layout)
                f1 = bb.emit_te(topi.layout_transform, f1, "OIHW", kernel_layout)
            if add_constraints:
                first_conv = bb.emit_te(
                    topi.nn.conv2d_NCHWc,
                    x,
                    f1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    layout="unused",
                    out_layout="unused",
                    primfunc_name_hint="conv2d",
                )
            else:
                first_conv = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    f1,
                    strides=1,
                    padding=0,
                    dilation=1,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    primfunc_name_hint="conv2d",
                )
            if add_constraints:
                first_conv = bb.emit_te(topi.layout_transform, first_conv, data_layout, "NCHW")
            add = bb.emit_te(topi.add, first_conv, bias)
            gv = bb.emit_output(add)
        bb.emit_func_output(gv)

    return bb.get()


def test_conv():
    print("\n\n========== INITIAL IR MODULE ==========\n")
    mod = relax_graph()
    mod.show()

    print("\n========== IR MODULE WITH LAYOUT CONSTRAINTS ==========\n")
    mod = relax_graph(add_constraints=True)
    first_conv = mod["first_conv"]
    second_conv = mod["second_conv"]
    mod["first_conv"] = first_conv.with_attr("layout", "frozen")
    mod["second_conv"] = second_conv.with_attr("layout", "frozen")
    mod.show()
    print("\n========== IR MODULE WITH LAYOUT CONSTRAINTS ==========\n")


def test_conv_tiny():
    print("\n\n========== INITIAL IR MODULE ==========\n")
    mod = relax_graph_tiny()
    mod.show()

    print("\n========== IR MODULE WITH LAYOUT CONSTRAINTS ==========\n")
    mod = relax_graph_tiny(add_constraints=True)
    mod.show()
    print("\n========== IR MODULE WITH LAYOUT CONSTRAINTS ==========\n")
    add = flow_constraint(
        mod["add"],
        "T_add",
        read_indices=[0],
        read_index_map=lambda N, C, H, W: (N, C // 4, H, W, C % 4),
    )
    add.show()
    add = flow_constraint(
        add,
        "T_add",
        read_indices=[1],
        write_indices=[],
        read_index_map=lambda C, H, W: (C // 4, H, W, C % 4),
    )

    add.show()


if __name__ == "__main__":
    pytest.main([__file__])
