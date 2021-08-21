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

from tvm import tir
from tvm import relax as rx


def test_irbuilder():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.IRBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.shape_[0] == m
            assert lv0.shape_[1] == n
            assert lv0.checked_type_.rank == 2
            assert lv0.checked_type_.dtype == "float16"
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            assert lv1.name_hint == "lv1"
            gv0 = ib.emit_df_output(lv1)
        assert gv0.name_hint == "gv0"
        assert gv0.shape_[0] == m
        assert gv0.shape_[1] == n
        assert gv0.checked_type_.rank == 2
        assert gv0.checked_type_.dtype == "float16"
        ib.emit_output(gv0)
    func = ib.get()
    assert func.params[0] == x
    assert func.name.name_hint == "func"
    assert func.body.body == gv0
    assert len(func.body.blocks) == 1
    assert len(func.body.blocks[0].bindings) == 3


if __name__ == "__main__":
    test_irbuilder()
