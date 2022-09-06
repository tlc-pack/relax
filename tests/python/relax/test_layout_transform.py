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
    print("\n** BEFORE **\n")
    print(before)
    if after:
        print("\n** AFTER **\n")
        print(after)
    if expected:
        print("\n** EXPECTED **\n")
        print(expected)


def print_relax(before, after=None, expected=None):
    print("\n** BEFORE **\n")
    before.show()
    if after:
        print("\n** AFTER **\n")
        after.show()
    if expected:
        print("\n** EXPECTED **\n")
        expected.show()


def test_reduce_op_convert_layout():
    # for reduce_op in [relay.argmax, relay.mean, relay.max]:

    for reduce_op in [relay.sum]:

        def before():
            x = relay.var("x", shape=(2, 64, 56, 56))
            weight = relay.var("weight", shape=(64, 64, 3, 3))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
            y = reduce_op(y, axis=[2, 3])
            y = relay.Function([x, weight], y)
            return y

        def expected():
            x = relay.var("x", shape=(2, 64, 56, 56))
            weight = relay.var("weight", shape=(64, 64, 3, 3))
            x = relay.layout_transform(x, "NCHW", "NHWC")
            weight = relay.layout_transform(weight, "OIHW", "HWIO")
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
            y = relay.Function(relay.analysis.free_vars(y), y)
            return y

        a = before()
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
        b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

        print_relay(before(), a, b)
        target = tvm.target.Target("llvm")
        relax_before = relay_translator.from_relay(before(), target)
        relax_after = relay_translator.from_relay(a, target)
        print_relax(relax_before["main"], relax_after["main"])

        print("\nLet's play with schedule\n")
        sch = tvm.tir.Schedule(relax_before["sum"])
        sch.transform_layout(
            "rxplaceholder_red",
            ("read", 0),
            index_map=lambda n, c0, h, w, c1: (n, h, w, c0 * 4 + c1),
        )
        sch.transform_layout(
            "rxplaceholder_red", ("write", 0), index_map=lambda i, j, k: (i, j * 4 + k)
        )
        # i = sch.get_loops("rxplaceholder_red")
        # sch.fuse(i[1], i[2])
        sch.mod.show()


def test_broadcast_layout():
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

    relay_before = before()
    relay_after = run_opt_pass(
        relay_before, transform.ConvertLayout({"nn.conv2d": ["NCHW4c", "OIHW4i4o"]})
    )
    print_relay(relay_before, relay_after)
    relax_before = relay_translator.from_relay(relay_before, "llvm")
    relax_after = relay_translator.from_relay(relay_after, "llvm")
    relax_before.show()
    print_relax(relax_before["add"], relax_after["add"])

    print("\nLet's play with schedule\n")

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

    add.show()
    sch = tvm.tir.Schedule(add)
    sch.transform_layout(
        "T_add",
        ("read", 0),
        index_map=lambda n, h, w, c: (n, c // 4, h, w, c % 4),
    )
    sch.transform_layout(
        "T_add",
        ("write", 0),
        index_map=lambda n, h, w, c: (n, c // 4, h, w, c % 4),
    )
    sch.mod.show()


def test_strided_slice_layout():
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
        y = relay.nn.relu(y)
        y = relay.strided_slice(y, begin=[0, 0, 0, 1], end=[1, 10, 56, -1], strides=[1, 2, 1, 1])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before = before()
    relay_after = run_opt_pass(
        relay_before, transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"]})
    )
    print_relay(relay_before, relay_after)
    relax_before = relay_translator.from_relay(relay_before, "llvm")
    relax_after = relay_translator.from_relay(relay_after, "llvm")
    # relax_before.show()
    print_relax(relax_before["strided_slice"], relax_after["strided_slice"])

    # print("\nLet's play with schedule\n")

    # @T.prim_func
    # def add(
    #     rxplaceholder: T.Buffer[(32, 56, 56, 64), "float32"],
    #     rxplaceholder_1: T.Buffer[(32, 56, 56, 1), "float32"],
    #     T_add: T.Buffer[(32, 56, 56, 64), "float32"],
    # ) -> None:
    #     # function attr dict
    #     T.func_attr({"global_symbol": "add", "tir.noalias": True})
    #     # body
    #     # with T.block("root")
    #     for i0, i1, i2, i3 in T.grid(32, 56, 56, 64):
    #         with T.block("T_add"):
    #             ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
    #             T.reads(rxplaceholder[ax0, ax1, ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, 0])
    #             T.writes(T_add[ax0, ax1, ax2, ax3])
    #             T_add[ax0, ax1, ax2, ax3] = (
    #                 rxplaceholder[ax0, ax1, ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, 0]
    #             )

    # add.show()
    # sch = tvm.tir.Schedule(add)
    # sch.transform_layout(
    #     "T_add",
    #     ("read", 0),
    #     index_map=lambda n, h, w, c: (n, c // 4, h, w, c % 4),
    # )
    # sch.transform_layout(
    #     "T_add",
    #     ("read", 1),
    #     index_map=lambda n, h, w, c: (n, c // 4, h, w, c % 4),
    # )
    # sch.mod.show()


if __name__ == "__main__":
    pytest.main([__file__])
