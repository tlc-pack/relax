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
import tvm
import tvm.testing
from tvm import relax
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder.base import IRBuilder


def test_function_simple():
    """
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        out = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        return out
    """
    # create with Script IRBuilder
    with IRBuilder() as ir_builder:
        with R.function():
            R.func_name("foo")
            R.func_attr({"Primitive": 1})
            x = R.arg("x", R.tensor((128, 128), "float32"))
            R.func_ret_type(R.tensor(dtype="float32", ndim=2))
            out = R.emit(R.call_tir("extern_func", x, (128, 128), dtype="float32"))
            IRBuilder.name("out", out)
            R.func_ret_value(out)
    func = ir_builder.get()
    # create with BlockBuilder
    x = relax.Var("x", [128, 128], relax.DynTensorType(2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": 1}):
        out = bb.emit(relax.call_tir("extern_func", x, (128, 128), dtype="float32"))
        bb.emit_func_output(out)
    mod = bb.get()

    tvm.ir.assert_structural_equal(func, mod["foo"])
    # check names
    assert func.attrs["global_symbol"] == "foo"
    assert func.params[0].name_hint == "x"
    assert func.body.body.name_hint == "out"


if __name__ == "__main__":
    tvm.testing.main()
