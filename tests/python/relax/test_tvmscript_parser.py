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

import pytest
from typing import Union
import tvm
import tvm.testing

from tvm import relax
from tvm import IRModule
from tvm.script.parser import ir as I, tir as T, relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Union[relax.Function, IRModule],
):
    # TODO(siyuan): add round-trip tests
    tvm.ir.assert_structural_equal(parsed, expect)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        R.func_attr({"Primitive": 1})
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        return gv0

    x = relax.Var("x", [128, 128], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": 1}):
        out = bb.emit(relax.call_tir("extern_func", x, (128, 128), dtype="float32"))
        bb.emit_func_output(out)

    _check(foo, bb.get()["foo"])


def test_error_report():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = gv1 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
            return gv0


def test_simple_module():
    @I.ir_module
    class TestModule:
        @T.prim_func
        def tir_func(x: T.Buffer((128, 128), "float32"), y: T.Buffer((128, 128), "float32")):
            T.func_attr({"global_symbol": "tir_func", "tir.noalias": True})
            for i, j in T.grid(128, 128):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            # TODO(Siyuan): Need to change to `TestModule.tir_func`
            gv0 = R.call_tir(tir_func, x, (128, 128), dtype="float32")
            return gv0

    x = relax.Var("x", [128, 128], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)

    _check(TestModule, bb.get())


if __name__ == "__main__":
    tvm.testing.main()
