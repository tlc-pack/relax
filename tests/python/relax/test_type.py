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
from __future__ import annotations  # must import to defer parsing of annotations
import numpy as np
import tvm
from tvm import relax as rx
from tvm import tir as tir
from tvm import te

import tvm.script
from tvm.script import tir as T, relax as R

def test_shape_type():
    t0 = rx.ShapeType()
    t1 = rx.ShapeType()
    assert t0 == t1

def test_dyn_tensor_type():
    t0 = rx.DynTensorType()
    assert t0.rank == -1
    t1 = rx.DynTensorType(3, "int32")
    assert t1.rank == 3
    assert t1.dtype == "int32"

def test_emit_te():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", [n, m], type_anno)
    y = rx.Var("y", [n, m], type_anno)
    z = rx.Var("z", [n, m], type_anno)

    def te_func(args, args_dict, msg):
        A, B = args
        C = args_dict["C"]
        D = te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
        E = te.compute((128, 128), lambda i, j: D[i, j] - C[i, j])
        return E

    with bb.function("rx_func", [x, y, z]):
        out = bb.emit_te(te_func, [x, y], {"C": z}, msg="hello")
        bb.emit_func_output(out)

    mod = bb.get()
    print(R.parser.astext(mod))


@T.prim_func
def identity_tir(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [54, 96], "float16")
    B = T.match_buffer(b, [54, 96], "float16")

    for i, j in T.grid(54, 96):
        with T.block("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


def test_call_tir_type() -> None:
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float16")
    v0 = rx.Var("v0", shape_anno, type_anno)
    v1 = rx.call_tir([54, 96], identity_tir, [v0])
    t0 = v1.checked_type
    assert t0.rank == 2
    assert t0.dtype == "float16"


if __name__ == "__main__":
    test_shape_type()
    test_dyn_tensor_type()
    test_call_tir_type()
