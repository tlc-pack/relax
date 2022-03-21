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
from tvm import relay
from tvm.script import tir as T

import numpy as np

def matmul():
  m = tir.Var("m", "int64")
  n = tir.Var("n", "int64")
  dtype0 = rx.DynTensorType(rank=2, dtype="float16")
  dtype1 = rx.DynTensorType(rank=1, dtype="float16")
  x = rx.Var("x", [m, n], dtype0)
  y = rx.Var("y", [n], dtype1)
  bb = rx.BlockBuilder()

  op = relay.op.get("multiply")

  with bb.function("func", [x, y]):
      with bb.dataflow() as df:
          #lv1 = bb.emit(rx.Call(op, [x, y]))
          gv0 = bb.emit_output(rx.Call(op, [x, y]))
          #gv0 = bb.emit_output(lv1)
      bb.emit_func_output(gv0)
  mod = bb.get()
  print(mod)

  new_mod = rx.transform.ReverseModeAD()(mod)
  # new_mod = rx.transform.EmitTERewrite()(mod)
  print(new_mod)
  exit()

  target = tvm.target.Target("llvm", host="llvm")
  ex, lib = rx.vm.build(new_mod, target)

  vm = rx.VirtualMachine(ex, tvm.cpu(), mod=lib)
  inp = tvm.nd.array(np.random.rand(5, 5).astype(np.float16))
  inp2 = tvm.nd.array(np.random.rand(5, ).astype(np.float16))
  res = vm["func"](inp, inp2)

  np.testing.assert_allclose(inp.numpy() * inp2.numpy(), res.numpy())

if __name__ == "__main__":
  matmul()
