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
import tvm.testing
from tvm import relax
import tvm.script
from tvm.script import tir as T, relax as R

# TODO (sunggg):
# Currently, parser sets each function name as the global symbol by default.
# And printer uses the global symbol to print out the function name.
# This is temproray and will improve in the future.


def check_if_func_exists(mod, func_name):
    gvs = [str(gv) for gv in mod.get_global_vars()]
    return ("@" + func_name) in gvs


def test_unused_relax_func():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_add(
            x: T.Buffer[(16, 16), "float32"],
            y: T.Buffer[(16, 16), "float32"],
            z: T.Buffer[(16, 16), "float32"],
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def unused_func(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_add, (x, w), (16, 16), dtype="float32")
            return gv0

    mod = InputModule
    assert mod
    # RemoveUnusedFunction pass won't remove the function with global symbol for the external reference.
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert check_if_func_exists(new_mod, "unused_func")

    # Remove global symbol from the function.
    mod["unused_func"] = mod["unused_func"].without_attr("global_symbol")

    # Then, this removes the unused function without any global symbol.
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_relax_func_custom_entry_func():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_add(
            x: T.Buffer[(16, 16), "float32"],
            y: T.Buffer[(16, 16), "float32"],
            z: T.Buffer[(16, 16), "float32"],
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def unused_func(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def foo(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.call_tir(tir_add, (x, w), (16, 16), dtype="float32")
            return gv0

    mod = InputModule
    assert mod

    # RemoveUnusedFunction pass won't remove the function with global symbol for the external reference.
    # Test entry function other than "main".
    new_mod = relax.transform.RemoveUnusedFunctions(entry_functions=["foo"])(mod)
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_add")
    assert check_if_func_exists(new_mod, "unused_func")

    # Remove global symbol from the function.
    mod["unused_func"] = mod["unused_func"].without_attr("global_symbol")

    # Then, this removes the unused function without any global symbol.
    new_mod = relax.transform.RemoveUnusedFunctions(entry_functions=["foo"])(mod)
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_relax_func_symbolic_shape():
    # Test with relax function w/ symbolic shape.
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_add(
            x: T.Buffer[(16, 16), "float32"],
            y: T.Buffer[(16, 16), "float32"],
            z: T.Buffer[(16, 16), "float32"],
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def unused_func(x: Tensor((m, n), "float32"), w: Tensor((n, k), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(x: Tensor((m, n), "float32"), w: Tensor((n, k), "float32")):
            gv0 = R.call_tir(tir_add, (x, w), (m, k), dtype="float32")
            return gv0

    mod = InputModule
    assert mod

    # RemoveUnusedFunction pass won't remove the function with global symbol for the external reference.
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert check_if_func_exists(new_mod, "unused_func")

    # Remove global symbol from the unused function
    mod["unused_func"] = mod["unused_func"].without_attr("global_symbol")
    # Remove unused function before shape lowering.
    # Test entry function other than "main".
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")

    # Remove unused function after shape lowering.
    # Shape lowering will inject several shape-related global functions.
    # We need to make sure unused function removal pass does not remove those functions.
    shape_lowered_mod = relax.transform.VMShapeLower()(mod)
    new_mod = relax.transform.RemoveUnusedFunctions()(shape_lowered_mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert check_if_func_exists(new_mod, "shape_func")  # injected by VMShapeLower pass
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_prim_func():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def unused_func(
            x: T.Buffer[(16, 16), "float32"],
            y: T.Buffer[(16, 16), "float32"],
            z: T.Buffer[(16, 16), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "tir_unused"})
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def relax_add(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = relax_add(x, w)
            return gv0

    mod = InputModule
    assert mod
    # RemoveUnusedFunction pass won't remove the function with global symbol for the external reference.
    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "relax_add")
    assert check_if_func_exists(new_mod, "unused_func")

    # Remove global symbol from the unused function
    mod["unused_func"] = mod["unused_func"].without_attr("global_symbol")

    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "relax_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_multiple_unused_funcs():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def unused_func1(
            x: T.Buffer[(16, 16), "float32"],
            y: T.Buffer[(16, 16), "float32"],
            z: T.Buffer[(16, 16), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "tir_unused"})
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def unused_func2(x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")):
            gv0 = relax.add(x, w)
            return gv0

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = relax.add(x, w)
            return gv0

    mod = InputModule
    assert mod

    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "unused_func1")
    assert check_if_func_exists(new_mod, "unused_func2")

    # Remove global symbol from unused functions
    mod["unused_func1"] = mod["unused_func1"].without_attr("global_symbol")
    mod["unused_func2"] = mod["unused_func2"].without_attr("global_symbol")

    new_mod = relax.transform.RemoveUnusedFunctions()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert not check_if_func_exists(new_mod, "unused_func1")
    assert not check_if_func_exists(new_mod, "unused_func2")


if __name__ == "__main__":
    pytest.main([__file__])
