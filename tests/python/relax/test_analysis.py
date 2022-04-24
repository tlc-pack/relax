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
import tvm
from tvm import tir
from tvm import relax as rx
from tvm.ir import structural_equal


def test_dispatch_var():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
    type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")
    v0 = rx.Var("v0", [m, n], type_anno0)
    v1 = rx.DataflowVar("v1", [n], type_anno1)
    t = None

    def fvisit(e):
        nonlocal t
        t = type(e)

    rx.analysis.post_order_visit(v0, fvisit)
    assert t == type(v0)
    rx.analysis.post_order_visit(v1, fvisit)
    assert t == type(v1)


def test_post_order_visit():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
    type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")
    x = rx.Var("x", [m, n], type_anno0)
    y = rx.Var("y", [n], type_anno1)
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    expr = ib.get()["func"]

    names = []

    def fvisit(e):
        nonlocal names
        if isinstance(e, tvm.ir.op.Op):
            names.append(e.name)

    rx.analysis.post_order_visit(expr.body, fvisit)
    assert names == ["relax.add", "relax.multiply"]


if __name__ == "__main__":
    pytest.main([__file__])
