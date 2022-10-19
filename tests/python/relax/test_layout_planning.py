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


data_transform = lambda N, H, W, C: (N, H // 8, W // 8, C // 32, H % 8, W % 8, C % 32)
filter_transform = lambda FH, FW, C, K: (K // 32, C // 32, FH, FW, (C % 32) // 4, K % 32, C % 4)




def test_conv_relax():
    def before():
        N, C, H, W = 4, 62, 52, 52
        K, FH, FW = 25, 3, 3
        x = relay.var("x", shape=(N, H, W, C))
        weight = relay.var("weight", shape=(FH, FW, C, K))
        y = relay.add(x, x)
        y = relay.nn.conv2d(
            y, weight, channels=K, kernel_size=(FH, FW), data_layout="NHWC", kernel_layout="HWIO"
        )
        y = relay.nn.relu(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    relay_before = before()
    print_relay(relay_before)

    relax_before = relay_translator.from_relay(
        relay_before, "llvm", disabled_pass=["AlterOpLayout"]
    )
    print_relax(relax_before)


if __name__ == "__main__":
    pytest.main([__file__])
