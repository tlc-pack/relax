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
  # m = tir.Var("m", "int64")
  # n = tir.Var("n", "int64")
  m = 5
  n = 5
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

  print(new_mod)
  # exit()
  target = tvm.target.Target("llvm", host="llvm")
  new_mod = rx.transform.EmitTERewrite(target)(new_mod)

  print(new_mod)

  ex = rx.vm.build(new_mod, target)

  vm = rx.VirtualMachine(ex, tvm.cpu())
  inp = tvm.nd.array(np.random.rand(5, 5).astype(np.float16))
  inp2 = tvm.nd.array(np.random.rand(5, ).astype(np.float16))
  res, res1, res2 = vm["func"](inp, inp2)

  # np.testing.assert_allclose(inp.numpy() * inp2.numpy(), res.numpy())

def test_nndense():
  m = 5
  n = 5
  dtype0 = rx.DynTensorType(rank=2, dtype="float16")
  dtype1 = rx.DynTensorType(rank=2, dtype="float16")
  x = rx.Var("x", [m, n], dtype0)
  y = rx.Var("y", [m, n], dtype1)
  bb = rx.BlockBuilder()

  # op = relay.op.get("nn.dense")

  with bb.function("func", [x, y]):
      with bb.dataflow() as df:
          #lv1 = bb.emit(rx.Call(op, [x, y]))
          lv1 = bb.emit(relay.nn.dense(x, y))
          # lv2 = bb.emit(relay.nn.relu(lv1))
          gv0 = bb.emit_output(relay.nn.relu(lv1))
          #gv0 = bb.emit_output(lv1)
      bb.emit_func_output(gv0)
  mod = bb.get()
  # print(mod)

  target = tvm.target.Target("llvm", host="llvm")
  new_mod = rx.transform.ReverseModeAD()(mod)
  print(new_mod)

  new_mod = rx.transform.EmitTERewrite(target)(new_mod)

  ex = rx.vm.build(new_mod, target)

  vm = rx.VirtualMachine(ex, tvm.cpu())
  inp = tvm.nd.array(np.random.uniform(low = -1, high = 1, size=(5, 5)).astype(np.float16))
  inp2 = tvm.nd.array(np.random.uniform(low = -1, high = 1, size=(5, 5)).astype(np.float16))
  res, res1, res2 = vm["func"](inp, inp2)

  # np.testing.assert_allclose(inp.numpy() @ (np.transpose(inp2.numpy())), res.numpy(), atol=1e-3, rtol=1e-3)
def test_logsoftmax():
  m = 5
  n = 5
  dtype0 = rx.DynTensorType(rank=2, dtype="float16")
  x = rx.Var("x", [m, n], dtype0)
  bb = rx.BlockBuilder()

  # op = relay.op.get("nn.dense")

  with bb.function("func", [x]):
      with bb.dataflow() as df:
          #lv1 = bb.emit(rx.Call(op, [x, y]))
          lv1 = bb.emit(relay.nn.log_softmax(x))
          # lv2 = bb.emit(relay.nn.relu(lv1))
          # gv0 = bb.emit_output(relay.nn.relu(lv1))
          gv0 = bb.emit_output(lv1)
      bb.emit_func_output(gv0)
  mod = bb.get()
  # print(mod)

  target = tvm.target.Target("llvm", host="llvm")
  new_mod = rx.transform.ReverseModeAD()(mod)
  print(new_mod)

  new_mod = rx.transform.EmitTERewrite(target)(new_mod)

  print(new_mod)

  ex = rx.vm.build(new_mod, target)

  vm = rx.VirtualMachine(ex, tvm.cpu())
  inp = tvm.nd.array(np.random.uniform(low = -1, high = 1, size=(5, 5)).astype(np.float16))
  res, res1 = vm["func"](inp)

def test_cross_entropy():
  m = 5
  n = 5
  dtype0 = rx.DynTensorType(rank=2, dtype="float16")
  dtype1 = rx.DynTensorType(rank=2, dtype="float16")
  x = rx.Var("x", [m, n], dtype0)
  y = rx.Var("y", [m, n], dtype1)
  bb = rx.BlockBuilder()

  with bb.function("func", [x, y]):
      with bb.dataflow() as df:
          #lv1 = bb.emit(rx.Call(op, [x, y]))
          gv0 = bb.emit_output(relay.nn.cross_entropy(x, y))
          #gv0 = bb.emit_output(lv1)
      bb.emit_func_output(gv0)
  mod = bb.get()
  print(mod)

  # new_mod = rx.transform.ReverseModeAD()(mod)

  # print(new_mod)
  # # exit()
  # target = tvm.target.Target("llvm", host="llvm")
  # new_mod = rx.transform.EmitTERewrite(target)(new_mod)

  # print(new_mod)

  # ex = rx.vm.build(new_mod, target)

  # vm = rx.VirtualMachine(ex, tvm.cpu())
  # inp = tvm.nd.array(np.random.rand(5, 5).astype(np.float16))
  # inp2 = tvm.nd.array(np.random.rand(5, ).astype(np.float16))
  # res, res1, res2 = vm["func"](inp, inp2)


if __name__ == "__main__":
  test_nndense()
  test_logsoftmax()
  test_cross_entropy()
  matmul()
