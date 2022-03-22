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
            ) -> Tensor[(10, 5), "float32"]:
                return relax.add(x2, y2)

            gv1: Tensor[(10, 5), "float32"] = inner(x1, y1)
            return gv1

    mod = TestBasic
    print(mod)
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 2
    assert_structural_equal(new_mod, TargetModule, map_free_vars=True)


def test_closure():
    # the target IRModule
    @tvm.script.ir_module
    class TargetModule:
        @R.function
        def lifted_func_0(y: Tensor[(), "float32"]):
            return lifted_func_1(y)

        @R.function
        def lifted_func_1(y1: Tensor[(), "float32"]):
            @R.function
            def inner_func(x: Tensor[(), "float32"]):
                return relax.add(x, y1)

            return inner_func

        @R.function
        def main():
            return lifted_func_0(relax.const(1.0, dtype="float32"))(
                relax.const(0.0, dtype="float32")
            )

    # IRModule to apply LambadaLift
    mod = tvm.IRModule()
    x = relax.Var(
        "x",
        type_annotation=relax.DynTensorType(0, "float32"),
    )
    y = relax.Var(
        "y",
        type_annotation=relax.DynTensorType(0, "float32"),
    )
    inner_func = relax.Function([x], relax.op.add(x, y), None, relax.GlobalVar("inner_func"))
    outer_func = relax.Function([y], inner_func, None, relax.GlobalVar("outer_func"))

    clo = outer_func(relax.const(1.0, dtype="float32"))

    mod["main"] = relax.Function(
        [],
        relax.Call(clo, [relax.const(0.0, dtype="float32")]),
        None,
        relax.GlobalVar("main"),
    )
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 3
    # assert_structural_equal(new_mod, TargetModule, map_free_vars=True)


def test_recursive():
    @tvm.script.ir_module
    class TestRecursive:
        @R.function
        def main(x: Tensor[(), "float32"]):
            @R.function
            def while_loop(i: Tensor[(), "int32"], s: Tensor[(), "float32"]):
                cond: Tensor[(), "bool"] = relax.call_packed("test.vm.less", i, relax.const(10))
                if cond:
                    new_i = relax.add(i, relax.const(1))
                    new_s = relax.add(s, x)
                    r: Tensor[(), "float32"] = while_loop(new_i, new_s)
                else:
                    r: Tensor[(), "float32"] = s
                return r

            gv: Tensor[(), "float32"] = while_loop(relax.const(0), relax.const(0))
            return gv

    mod = TestRecursive
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 2


if __name__ == "__main__":
    pytest.main([__file__])
