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
from tvm import relax
from tvm.runtime.object import Object
import tvm.script
from tvm.script import relax as R, tir as T
from tvm.relax import transform
from tvm.ir.base import assert_structural_equal


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
        def lifted_func_0(x2: Tensor((10, 5), "float32"), y2: Tensor((10, 5), "float32")) -> Tensor:
            s: Tensor((10, 5), "float32") = relax.add(x2, y2)
            return s

        @R.function
        def main(
            x1: Tensor((10, 5), "float32"), y1: Tensor((10, 5), "float32")
        ) -> Tensor((10, 5), "float32"):
            inner = lifted_func_0
            gv1 = inner(x1, y1)
            return gv1

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x1: Tensor((10, 5), "float32"), y1: Tensor((10, 5), "float32")
        ) -> Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: Tensor((10, 5), "float32"), y2: Tensor((10, 5), "float32")
            ) -> Tensor((10, 5), "float32"):
                s: Tensor((10, 5), "float32") = relax.add(x2, y2)
                return s

            gv1: Tensor((10, 5), "float32") = inner(x1, y1)
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
        def main(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")):
            outer_func = lifted_func_0
            in_call = outer_func(x)
            res = relax.invoke_closure(in_call, (y,), type_args=(Tensor(ndim=2, dtype="float32")))
            return res

        @R.function
        def lifted_func_1(x1: Tensor((2, 3), "float32"), c1: Tensor((2, 3), "float32")):
            r_1: Tensor((2, 3), "float32") = relax.add(x1, c1)
            return r_1

        @R.function
        def lifted_func_0(y: Tensor((2, 3), "float32")):
            return relax.make_closure(lifted_func_1, (y,))

    # IRModule to perform Lambda Lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")
        ) -> Tensor((2, 3), "float32"):
            @R.function
            def outer_func(c1: Tensor((2, 3), "float32")):
                @R.function
                def inner_func(x1: Tensor((2, 3), "float32")):
                    s: Tensor((2, 3), "float32") = relax.add(x1, c1)
                    return s

                return inner_func

            in_call = outer_func(x)
            res = in_call(y)
            return res

    before = Before
    after = transform.LambdaLift()(before)
    expected = Expected
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_recursive():
    # the expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def lifted_func_0(
            i: Tensor((), "int32"), s: Tensor((2, 3), "float32"), x: Tensor((2, 3), "float32")
        ) -> Tensor((2, 3), "float32"):
            cond: Tensor((), "bool") = relax.call_packed(
                "test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool"))
            )
            c: Tensor((), "int32") = relax.const(1, dtype="int32")
            if cond:
                new_i: Tensor((), "int32") = relax.add(i, c)
                new_s: Tensor((2, 3), "float32") = relax.add(s, x)
                r = lifted_func_0(new_i, new_s, x)
            else:
                r = s
            return r

        @R.function
        def main(x: Tensor((2, 3), "float32")) -> Tensor:
            while_loop = relax.make_closure(lifted_func_0, (x,))
            gv = relax.invoke_closure(
                while_loop, (relax.const(0), x), type_args=(Tensor(ndim=2, dtype="float32"))
            )
            return gv

    # the IRModule to apply lambda lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((2, 3), "float32")) -> Tensor:
            @R.function
            def while_loop(
                i: Tensor((), "int32"), s: Tensor((2, 3), "float32")
            ) -> Tensor((2, 3), "float32"):
                cond: Tensor((), "bool") = relax.call_packed(
                    "test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool"))
                )
                c: Tensor((), "int32") = relax.const(1, dtype="int32")
                if cond:
                    new_i: Tensor((), "int32") = relax.add(i, c)
                    new_s: Tensor((2, 3), "float32") = relax.add(s, x)
                    r: Tensor((2, 3), "float32") = while_loop(new_i, new_s)
                else:
                    r: Tensor((2, 3), "float32") = s
                return r

            gv: Tensor((2, 3), "float32") = while_loop(relax.const(0), x)
            return gv

    before = Before
    expected = Expected
    # Perform Lamda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_multi_func():
    # expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def glob_func_1(
            x1: Tensor((10, 5), "float32"), y1: Tensor((10, 5), "float32")
        ) -> Tensor(None, "float32", ndim=2):
            inner = lifted_func_1
            gv1 = inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x11: Tensor((10, 5), "float32"), y11: Tensor((10, 5), "float32")
        ) -> Tensor(None, "float32", ndim=2):
            inner1 = lifted_func_0
            gv11 = inner1(x11, y11)
            return gv11

        @R.function
        def lifted_func_0(
            x2: Tensor((10, 5), "float32"), y2: Tensor((10, 5), "float32")
        ) -> Tensor(None, "float32", ndim=2):
            s: Tensor((10, 5), "float32") = relax.add(x2, y2)
            return s

        @R.function
        def lifted_func_1(
            x21: Tensor((10, 5), "float32"), y21: Tensor((10, 5), "float32")
        ) -> Tensor(None, "float32", ndim=2):
            s1: Tensor((10, 5), "float32") = relax.add(x21, y21)
            return s1

    # the IRModule to apply lambda lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def glob_func_1(
            x1: Tensor((10, 5), "float32"), y1: Tensor((10, 5), "float32")
        ) -> Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: Tensor((10, 5), "float32"), y2: Tensor((10, 5), "float32")
            ) -> Tensor((10, 5), "float32"):
                s: Tensor((10, 5), "float32") = relax.add(x2, y2)
                return s

            gv1: Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x1: Tensor((10, 5), "float32"), y1: Tensor((10, 5), "float32")
        ) -> Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: Tensor((10, 5), "float32"), y2: Tensor((10, 5), "float32")
            ) -> Tensor((10, 5), "float32"):
                s: Tensor((10, 5), "float32") = relax.add(x2, y2)
                return s

            gv1: Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lamda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 4
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_no_local_func():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def sub(
            A: T.Buffer[(16, 16), "float32"],
            B: T.Buffer[(16, 16), "float32"],
            C: T.Buffer[(16, 16), "float32"],
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("sub"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: Tensor((16, 16), "float32"), x: Tensor((_, _), "float32")):
            s = relax.call_tir(sub, (c0, x), (16, 16), dtype="float32")
            return s

    before = Before
    # Perform lambda lifting
    after = transform.LambdaLift()(before)
    # No local functions are lifted
    assert_structural_equal(after, before, map_free_vars=True)
    _check_save_roundtrip(after)


if __name__ == "__main__":
    pytest.main((__file__))
