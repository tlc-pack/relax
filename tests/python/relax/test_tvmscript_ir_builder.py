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
from tvm import relax, tir
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


def test_match_shape():
    """
    @R.function
    def foo(x: R.Tensor(None, "float32"), y: R.Tensor(None, "float32")):
        m = T.var("int64")
        n = T.var("int64")
        R.match_shape(x, (m,))
        y1 = R.match_shape(x, (n,))
        return (m, n * 2)
    """
    # create with Script IRBuilder
    with IRBuilder() as ir_builder:
        with R.function():
            R.func_name("foo")
            x = R.arg("x", R.tensor(ndim=-1, dtype="float32"))
            y = R.arg("y", R.tensor(ndim=-1, dtype="float32"))
            m = tir.Var("m", dtype="int64")
            n = tir.Var("n", dtype="int64")
            R.emit_match_shape(x, (m,), emit_var=False)
            y1 = R.emit_match_shape(y, (n,), emit_var=True)
            IRBuilder.name("y1", y1)
            R.func_ret_value(relax.ShapeExpr([m, n * 2]))
    func = ir_builder.get()

    # create with BlockBuilder
    x = relax.Var("x", type_annotation=relax.DynTensorType(-1, "float32"))
    y = relax.Var("y", type_annotation=relax.DynTensorType(-1, "float32"))
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        bb.match_shape_binding(relax.MatchShape(x, (m,), var=None))
        y1 = bb.match_shape(y, (n,))
        bb.emit_func_output(relax.ShapeExpr([m, n * 2]))
    mod = bb.get()

    tvm.ir.assert_structural_equal(func, mod["foo"])


if __name__ == "__main__":
    tvm.testing.main()
