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


data_transform = lambda N, C, H, W: (N, H // 8, W // 8, C // 32, H % 8, W % 8, C % 32)
filter_transform = lambda K, C, H, W: (K // 32, C // 32, H, W, (C % 32) // 4, K % 32, C % 4)


def before():
    x = relay.var("x", shape=(1, 64, 43, 37))
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    y = relay.nn.conv2d(
        x, weight, channels=64, kernel_size=(3, 3), data_layout="NCHW", kernel_layout="OIHW"
    )
    y = relay.Function(analysis.free_vars(y), y)
    return y


@pytest.mark.xfail
def test_conv_relay():

    relay_before, relay_after, relax_before, relax_after = apply_conv_layout_conversion(
        before(), ["NHWC8h8w32c", "OIHW8i32o4i"]
    )


@tvm.script.ir_module
class ConvModule:
    @T.prim_func
    def conv2d(
        rxplaceholder: T.Buffer[(1, 64, 43, 37), "float32"],
        rxplaceholder_1: T.Buffer[(64, 64, 3, 3), "float32"],
        output_unpack: T.Buffer[(1, 64, 41, 35), "float32"],
    ) -> None:
        data_vec = T.alloc_buffer([1, 16, 43, 37, 4], dtype="float32")
        kernel_vec = T.alloc_buffer([16, 16, 3, 3, 4, 4], dtype="float32")
        conv2d_NCHWc = T.alloc_buffer([1, 16, 41, 35, 4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 16, 43, 37, 4):
            with T.block("data_vec"):
                bs, c, h, w, vc = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(rxplaceholder[bs, c * 4 + vc, h, w])
                T.writes(data_vec[bs, c, h, w, vc])
                data_vec[bs, c, h, w, vc] = rxplaceholder[bs, c * 4 + vc, h, w]
        for i0, i1, i2, i3, i4, i5 in T.grid(16, 16, 3, 3, 4, 4):
            with T.block("kernel_vec"):
                occ, icc, k_h, k_w, icb, ocb = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                T.reads(rxplaceholder_1[occ * 4 + ocb, icc * 4 + icb, k_h, k_w])
                T.writes(kernel_vec[occ, icc, k_h, k_w, icb, ocb])
                kernel_vec[occ, icc, k_h, k_w, icb, ocb] = rxplaceholder_1[
                    occ * 4 + ocb, icc * 4 + icb, k_h, k_w
                ]
        for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(1, 16, 41, 35, 4, 64, 3, 3):
            with T.block("conv2d_NCHWc"):
                n, oc_chunk, oh, ow, oc_block, ic, kh, kw = T.axis.remap(
                    "SSSSSRRR", [i0, i1, i2, i3, i4, i5, i6, i7]
                )
                T.reads(
                    data_vec[n, ic // 4, oh + kh, ow + kw, ic % 4],
                    kernel_vec[oc_chunk, ic // 4, kh, kw, ic % 4, oc_block],
                )
                T.writes(conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block])
                T.block_attr(
                    {
                        "workload": [
                            "conv2d_NCHWc.x86",
                            ["TENSOR", [1, 64, 43, 37], "float32"],
                            ["TENSOR", [64, 64, 3, 3], "float32"],
                            [1, 1],
                            [0, 0, 0, 0],
                            [1, 1],
                            "NCHW",
                            "NCHW",
                            "float32",
                        ]
                    }
                )
                with T.init():
                    conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = (
                    conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block]
                    + data_vec[n, ic // 4, oh + kh, ow + kw, ic % 4]
                    * kernel_vec[oc_chunk, ic // 4, kh, kw, ic % 4, oc_block]
                )
        for i0, i1, i2, i3 in T.grid(1, 64, 41, 35):
            with T.block("output_unpack"):
                n, c, h, w = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_NCHWc[n, c // 4, h, w, c % 4])
                T.writes(output_unpack[n, c, h, w])
                output_unpack[n, c, h, w] = conv2d_NCHWc[n, c // 4, h, w, c % 4]

    @R.function
    def main(
        x: Tensor((1, 64, 43, 37), "float32"), weight: Tensor((64, 64, 3, 3), "float32")
    ) -> Tensor(None, "float32", ndim=4):
        # block 0
        with R.dataflow():
            lv = R.call_tir(conv2d, (x, weight), (1, 64, 41, 35), dtype="float32")
            gv: Tensor((1, 64, 41, 35), "float32") = lv
            R.output(gv)
        return gv


def test_conv_relax():

    relay_before = before()
    print_relay(relay_before)

    relax_before = relay_translator.from_relay(
        relay_before, "llvm", disabled_pass=["AlterOpLayout"]
    )
    print_relax(relax_before)
    relax_before = ConvModule
    relax_before.show()


def test_conv_crouton_relax():
    def before():
        x = relay.var("x", shape=(32, 50, 50, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x, weight, channels=64, kernel_size=(3, 3), data_layout="NHWC", kernel_layout="HWIO"
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before = before()

    print_relay(relay_before)
    relax_before = relay_translator.from_relay(
        relay_before, "llvm", disabled_pass=["AlterOpLayout"]
    )
    print_relax(relax_before["conv2d"])

    relax_before = ConvModule
    print_relax(relax_before["conv2d"])
    layout_transform = lambda N, H, W, C: (N, H // 8, W // 8, C // 32, H % 8, W % 8, C % 32)


if __name__ == "__main__":
    pytest.main([__file__])
