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
import sys
import tvm
from tvm import relax as rx, tir
from tvm.script import tir as T, relax as R


def _check_equal(x, y):
    tvm.ir.assert_structural_equal(x, y)
    tvm.ir.assert_structural_equal(y, x)

    xhash = tvm.ir.structural_hash(x)
    yhash = tvm.ir.structural_hash(y)

    assert xhash == yhash


def _check_save_roundtrip(x):
    y = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, y)


def test_var_binding():
    dtype = rx.DynTensorType(1)
    x = rx.Var("x", [10], dtype)
    y = rx.Var("y", [10], dtype)

    def generator(x, y):
        bb = rx.BlockBuilder()
        bb._begin_binding_block()
        bb.emit(rx.op.add(x, y))
        return bb._end_block()

    block0 = generator(x, y)
    block1 = generator(x, y)

    _check_equal(block0, block1)


def test_match_shape():
    dtype = rx.DynTensorType(1)
    x = rx.Var("x", [10], dtype)
    m = tir.Var("m", dtype="int32")

    def generator(x):
        bb = rx.BlockBuilder()
        bb._begin_binding_block()
        bb.match_shape(x, [m * 2])
        return bb._end_block()

    block0 = generator(x)
    block1 = generator(x)

    _check_equal(block0, block1)


def test_function():
    def generator():
        dtype = rx.DynTensorType(1, "float32")
        x = rx.Var("x", [10], dtype)
        y = rx.Var("y", [10], dtype)
        bb = rx.BlockBuilder()
        with bb.function("name", [x, y]):
            gv = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv)
        return bb.get()

    func0 = generator()
    func1 = generator()
    _check_equal(func0, func1)


def test_ir_module():
    def generator():
        dtype = rx.DynTensorType(1, "float32")
        bb = rx.BlockBuilder()
        x = rx.Var("x", [10], dtype)
        y = rx.Var("y", [10], dtype)
        with bb.function("test", [x, y]):
            gv = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv)

        # get global var
        func_gv = bb.get().get_global_var("test")

        x = rx.Var("x", [10], dtype)
        y = rx.Var("y", [10], dtype)
        with bb.function("main", [x, y]):
            gv = bb.emit(rx.Call(func_gv, [x, y]))
            bb.emit_func_output(gv)
        return bb.get()

    mod0 = generator()
    mod1 = generator()
    _check_equal(mod0, mod1)


def test_match_shape_symbolic():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def f(x: Tensor[(_, _), "float32"]):
            x0 = R.match_shape(x, (n, m))
            return (x0, (n + 1, m))

    _check_save_roundtrip(InputModule)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
