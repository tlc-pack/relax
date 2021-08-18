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
from tvm import relay
from tvm.relay import op
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
            lv0 = ib.emit(relay.Call(op.get("relax.add"), [x, y]))
            lv1 = ib.emit(relay.Call(op.get("relax.multiply"), [lv0, y]))
            gv0 = ib.emit_df_output(lv1)
        ib.emit_output(gv0)
    func = ib.get()


if __name__ == "__main__":
    test_irbuilder()
