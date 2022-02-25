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
from tvm.script import tir as T


@tvm.register_func("test.op.identity")
def identity_packed(a):
    return tvm.nd.array(a.asnumpy())


@T.prim_func
def identity_tir(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [54, 96])
    B = T.match_buffer(b, [54, 96])

    for i, j in T.grid(54, 96):
        with T.block("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


def test_call_tir() -> None:
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float32")
    v0 = rx.Var("v0", shape_anno, type_anno)
    v1 = rx.call_tir([54, 96], rx.extern("test.op.identity"), [v0])
    v1 = rx.call_tir([54, 96], identity_tir, [v0])


if __name__ == "__main__":
    pytest.main([__file__])
