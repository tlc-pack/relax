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
from tvm.relay import op
from tvm import relax as rx
from tvm.ir import TensorType


def test_irbuilder():
    ib = rx.ir_builder.create()
    shape_anno = [24, 56]
    type_anno = TensorType(shape_anno, "float16")
    x = ib.var("x", shape_anno, type_anno)
    y = ib.var("y", shape_anno, type_anno)
    with ib.function("func", [x, y]):
        with ib.dataflow():
            z = ib.dataflow_var("y", shape_anno, type_anno)
            ib.bind(z, ib.call(op.op.get("add"), [x, y]))
            res = ib.var("res")
            ib.bind(res, ib.call(op.op.get("multiply"), [x, z]))
        ib.output(res)
    func = ib.get()
    assert func.params[0] == x
    assert func.name.name_hint == "func"
    assert func.body.body == res


if __name__ == "__main__":
    test_irbuilder()
