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


if __name__ == "__main__":
    tvm.testing.main()
