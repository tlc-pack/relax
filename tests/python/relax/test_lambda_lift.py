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


def test_basic():
    # the target IRModule
    @tvm.script.ir_module
    class TargetModule:
        @R.function
        def lifted_func_0(
            x2: Tensor[(10, 5), "float32"], y2: Tensor[(10, 5), "float32"]
        ) -> Tensor[(_, _), "float32"]:
            return relax.add(x2, y2)

        @R.function
        def main(x1: Tensor[(10, 5), "float32"], y1: Tensor[(10, 5), "float32"]):
            # block 0
            inner = lifted_func_0
            gv1 = inner(x1, y1)
            return gv1

    @tvm.script.ir_module
    class TestBasic:
        @R.function
        def main(x1: Tensor[(10, 5), "float32"], y1: Tensor[(10, 5), "float32"]):
            @R.function
            def inner(
                x2: Tensor[(10, 5), "float32"], y2: Tensor[(10, 5), "float32"]
            ) -> Tensor[(_, _), "float32"]:
                return relax.add(x2, y2)

            gv1: Tensor[(10, 5), "float32"] = inner(x1, y1)
            return gv1

    mod = TestBasic

    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 2
    assert_structural_equal(new_mod, TargetModule, map_free_vars=True)


def test_closure():
    # the target IRModule
    @tvm.script.ir_module
    class TargetModule:
        @R.function
        def lifted_func_0(y: Tensor[(2, 3), "float32"]):
            inner_func = lifted_func_1(y)
            return inner_func

        @R.function
        def lifted_func_1(y1: Tensor[(2, 3), "float32"]):
            @R.function
            def inner_func_0(x1: Tensor[(2, 3), "float32"]):
                r_1: Tensor[(2, 3), "float32"] = relax.add(x1, y1)
                return r_1

            return inner_func_0

        @R.function
        def main(x: Tensor[(2, 3), "float32"], y: Tensor[(2, 3), "float32"]):
            outer_func = lifted_func_0
            res = outer_func(x)(y)
            return res

    # IRModule to apply LambadaLift
    @tvm.script.ir_module
    class TestClosure:
        @R.function
        def main(x: Tensor[(2, 3), "float32"], y: Tensor[(2, 3), "float32"]):
            @R.function
            def outer_func(y1: Tensor[(2, 3), "float32"]):
                # c_1: Tensor[(), "float32"] = relax.const(3.14, dtype="float32")

                @R.function
                def inner_func(x1: Tensor[(2, 3), "float32"]):
                    r_1: Tensor[(2, 3), "float32"] = relax.add(x1, y1)
                    # s_2: Tensor[(), "float32"] = relax.add(s_1, c_1)
                    return r_1

                return inner_func

            res = outer_func(x)(y)
            # res = func(y)
            return res

    mod = TestClosure
    new_mod = transform.LambdaLift()(mod)
    assert_structural_equal(new_mod, TargetModule, map_free_vars=True)


def test_recursive():
    # the target IRModule
    @tvm.script.ir_module
    class TargetModule:
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

    @tvm.script.ir_module
    class TestRecursive:
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

    mod = TestRecursive
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 2
    assert_structural_equal(new_mod, TargetModule, map_free_vars=True)


if __name__ == "__main__":
    pytest.main([__file__])
