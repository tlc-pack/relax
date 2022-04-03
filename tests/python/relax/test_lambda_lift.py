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
import pytest
import tvm
from tvm import relax, tir, te
import tvm.script
from tvm.script import tir as T, relax as R
from tvm.relax import transform
from tvm.ir.base import assert_structural_equal, structural_equal


def _check_equal(x, y):
    tvm.ir.assert_structural_equal(x, y)
    tvm.ir.assert_structural_equal(y, x)

    xhash = tvm.ir.structural_hash(x)
    yhash = tvm.ir.structural_hash(y)

    assert xhash == yhash


def _check_save_roundtrip(x):
    y = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, y)


def test_basic():
    # the target IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def lifted_func_0(
            x2: Tensor[(10, 5), "float32"], y2: Tensor[(10, 5), "float32"]
        ) -> Tensor[(_, _), "float32"]:
            return relax.add(x2, y2)

        @R.function
        def main(x1: Tensor[(10, 5), "float32"], y1: Tensor[(10, 5), "float32"]):
            inner = lifted_func_0
            gv1 = inner(x1, y1)
            return gv1

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x1: Tensor[(10, 5), "float32"], y1: Tensor[(10, 5), "float32"]):
            @R.function
            def inner(
                x2: Tensor[(10, 5), "float32"], y2: Tensor[(10, 5), "float32"]
            ) -> Tensor[(_, _), "float32"]:
                return relax.add(x2, y2)

            gv1: Tensor[(10, 5), "float32"] = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_closure():
    # the expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor[(2, 3), "float32"], y: Tensor[(2, 3), "float32"]):
            outer_func = lifted_func_0
            res = outer_func(x)(y)
            return res

        @R.function
        def lifted_func_1(y1: Tensor[(2, 3), "float32"]):
            @R.function
            def inner_func_0(x1: Tensor[(2, 3), "float32"]):
                r_1: Tensor[(2, 3), "float32"] = relax.add(x1, y1)
                return r_1

            return inner_func_0

        @R.function
        def lifted_func_0(y: Tensor[(2, 3), "float32"]):
            return lifted_func_1(y)

    # IRModule to perform Lambda Lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor[(2, 3), "float32"], y: Tensor[(2, 3), "float32"]):
            @R.function
            def outer_func(c1: Tensor[(2, 3), "float32"]):
                @R.function
                def inner_func(x1: Tensor[(2, 3), "float32"]):
                    r_1: Tensor[(2, 3), "float32"] = relax.add(x1, c1)
                    return r_1

                return inner_func

            res = outer_func(x)(y)
            return res

    before = Before
    expected = Expected
    after = transform.LambdaLift()(before)
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_recursive():
    # the expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def lifted_func_0(x: Tensor[(2, 3), "float32"]):
            @R.function
            def inner_func_0(i: Tensor[(), "int32"], s: Tensor[(2, 3), "float32"]):
                cond = relax.call_packed("test.vm.less", i, relax.const(10))
                c: Tensor[(), "int32"] = relax.const(1, dtype="int32")
                if cond:
                    new_i: Tensor[(), "int32"] = relax.add(i, c)
                    new_s: Tensor[(2, 3), "float32"] = relax.add(s, x)
                    r = lifted_func_0(x)(new_i, new_s)
                else:
                    r = s
                return r

            return inner_func_0

        @R.function
        def main(x: Tensor[(2, 3), "float32"]):
            while_loop = lifted_func_0(x)
            gv = while_loop(relax.const(0), x)
            return gv

    # the IRModule to apply lambda lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor[(2, 3), "float32"]):
            @R.function
            def while_loop(i: Tensor[(), "int32"], s: Tensor[(2, 3), "float32"]):
                cond: Tensor[(), "bool"] = relax.call_packed("test.vm.less", i, relax.const(10))
                c: Tensor[(), "int32"] = relax.const(1, dtype="int32")
                if cond:
                    new_i: Tensor[(), "int32"] = relax.add(i, c)
                    new_s: Tensor[(2, 3), "float32"] = relax.add(s, x)
                    r: Tensor[(2, 3), "float32"] = while_loop(new_i, new_s)
                else:
                    r: Tensor[(2, 3), "float32"] = s
                return r

            gv: Tensor[(), "float32"] = while_loop(relax.const(0), x)
            return gv

    before = Before
    expected = Expected
    # Perform Lamda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after["lifted_func_0"], expected["lifted_func_0"], map_free_vars=True)
    _check_save_roundtrip(after)


if __name__ == "__main__":
    pytest.main([__file__])
