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

from typing import Optional, Union, Callable

import tvm
import tvm.testing
from tvm import IRModule, relax
from tvm.script.parser import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.parse(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def _test_unary(op_func: Callable):
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = op_func(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(op_func(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def _test_binary_arith(op_func: Callable):
    @R.function
    def foo(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = op_func(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y]):
        gv = bb.emit(op_func(x, y))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def _test_binary_cmp(op_func: Callable):
    @R.function
    def foo(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "bool"):
        gv: R.Tensor((2, 3), "bool") = op_func(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y]):
        gv = bb.emit(op_func(x, y))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_unary():
    _test_unary(relax.op.cos)
    _test_unary(relax.op.exp)
    _test_unary(relax.op.log)
    _test_unary(relax.op.negative)
    _test_unary(relax.op.sigmoid)
    _test_unary(relax.op.sin)
    _test_unary(relax.op.sqrt)
    _test_unary(relax.op.tanh)


def test_binary_arith():
    _test_binary_arith(relax.op.add)
    _test_binary_arith(relax.op.divide)
    _test_binary_arith(relax.op.floor_divide)
    _test_binary_arith(relax.op.multiply)
    _test_binary_arith(relax.op.subtract)


def test_binary_cmp():
    _test_binary_cmp(relax.op.equal)
    _test_binary_cmp(relax.op.greater)
    _test_binary_cmp(relax.op.greater_equal)
    _test_binary_cmp(relax.op.less)
    _test_binary_cmp(relax.op.less_equal)
    _test_binary_cmp(relax.op.not_equal)


def test_relax_ewise_fma():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 4), dtype="float32"),
        y: R.Tensor((2, 3, 4), dtype="float32"),
        z: R.Tensor((2, 3, 4), dtype="float32"),
    ) -> R.Tensor((2, 3, 4), dtype="float32"):
        gv: R.Tensor((2, 3, 4), dtype="float32") = R.ewise_fma(x, y, z)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))
    z = relax.Var("z", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y, z]):
        gv = bb.emit(relax.op.ewise_fma(x, y, z))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
