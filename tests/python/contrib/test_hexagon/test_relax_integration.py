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

import numpy as np
import pytest
import tvm.testing
from tvm import relay, relax, runtime
from tvm.relax.testing import relay_translator
from tvm.contrib.hexagon.session import Session
from tvm.script import relax as R, tir as T
from tvm.relay import testing


@tvm.testing.requires_hexagon
def test_conv2d(hexagon_session: Session):
    dtype = "float32"
    data = relay.var("data", relay.TensorType((1, 64, 64, 3), dtype))
    weight = relay.var("weight", relay.TensorType((5, 5, 3, 8), dtype))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    relay_mod = tvm.IRModule.from_expr(f)

    # target_hexagon = "llvm -keys=hexagon -link-params=0 -mattr=+hvxv69,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp -mcpu=hexagonv69 -mtriple=hexagon"
    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target)

    ex = relax.vm.build(relax_mod, target)
    dev = hexagon_session.device
    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, dev)

    data_np = np.random.rand(1, 64, 64, 3).astype(np.float32)
    weight_np = np.random.rand(5, 5, 3, 8).astype(np.float32)

    # Run on hexagon and get result
    data = tvm.nd.array(data_np, dev)
    weight = tvm.nd.array(weight_np, dev)
    hexagon_res = vm_rt["main"](data, weight)

    # Compile and run on Relay for comparison.
    dev = tvm.cpu()
    data = tvm.nd.array(data_np, dev)
    weight = tvm.nd.array(weight_np, dev)

    target = tvm.target.Target("llvm", host="llvm")
    vm_exec = relay.vm.compile(relay_mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    relay_res = vm_factory.invoke("main", data, weight)
    tvm.testing.assert_allclose(hexagon_res.numpy(), relay_res.numpy(), rtol=1e-3)


@tvm.testing.requires_hexagon
def test_conv2d_dyn(hexagon_session: Session):
    dtype = "float32"
    data = relay.var("data", relay.TensorType((relay.Any(), 64, 64, 3), dtype))
    weight = relay.var("weight", relay.TensorType((5, 5, 3, 8), dtype))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    relay_mod = tvm.IRModule.from_expr(f)

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target)

    ex = relax.vm.build(relax_mod, target)
    hexgaon_device = hexagon_session.device
    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, hexgaon_device)

    data_np = np.random.rand(1, 64, 64, 3).astype(np.float32)
    weight_np = np.random.rand(5, 5, 3, 8).astype(np.float32)

    # Run on hexagon and get result
    data = tvm.nd.array(data_np, hexgaon_device)
    weight = tvm.nd.array(weight_np, hexgaon_device)
    vm_rt.set_input("main", data, weight)
    hexagon_res = vm_rt["main"]()

    # Compile and run on Relay for comparison.
    cpu_device = tvm.cpu()
    data = tvm.nd.array(data_np, cpu_device)
    weight = tvm.nd.array(weight_np, cpu_device)

    target = tvm.target.Target("llvm", host="llvm")
    vm_exec = relay.vm.compile(relay_mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    relay_res = vm_factory.invoke("main", data, weight)
    tvm.testing.assert_allclose(hexagon_res.numpy(), relay_res.numpy(), rtol=1e-3)


@tvm.testing.requires_hexagon
def test_mlp(hexagon_session: Session):
    relay_mod, params = testing.mlp.get_workload(batch_size=1, dtype="float32")

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)

    ex = relax.vm.build(relax_mod, target)
    hexagon_device = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, hexagon_device)

    shape = (1, 1, 28, 28)
    data_np = np.random.rand(*shape).astype("float32")
    data = tvm.nd.array(data_np, hexagon_device)
    hexagon_res = vm_rt["main"](data)

    # Compile and run on Relay for comparison.
    cpu_dev = tvm.cpu()
    data = tvm.nd.array(data_np, cpu_dev)

    target = tvm.target.Target("llvm", host="llvm")
    vm_exec = relay.vm.compile(relay_mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, cpu_dev)
    relay_res = vm_factory.invoke("main", data, **params)
    tvm.testing.assert_allclose(hexagon_res.numpy(), relay_res.numpy(), rtol=1e-3)


@tvm.testing.requires_hexagon
def test_mlp_dyn(hexagon_session: Session):
    relay_mod, params = testing.mlp.get_workload(batch_size=relay.Any(), dtype="float32")
    shape = (1, 1, 28, 28)
    data_np = np.random.rand(*shape).astype("float32")

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    # translate the relay mobilenet and bind params
    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)

    # Compile and run on Hexagon.
    ex = relax.vm.build(relax_mod, target)
    dev = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, dev)
    data = tvm.nd.array(data_np, dev)
    vm_rt.set_input("main", data)
    hexagon_res = vm_rt["main"]()

    # Compile and run on Relay for comparison.
    dev = tvm.cpu()
    data = tvm.nd.array(data_np, dev)

    target = tvm.target.Target("llvm", host="llvm")
    vm_exec = relay.vm.compile(relay_mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, dev)
    relay_res = vm_factory.invoke("main", data, **params)
    print(hexagon_res)
    print(relay_res)
    tvm.testing.assert_allclose(hexagon_res.numpy(), relay_res.numpy(), rtol=1e-3)


def get_onnx_mobilenet():
    """Download and import mobilenet model with ONNX"""
    import onnx  # pylint: disable=import-outside-toplevel

    model_url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    model_path = tvm.contrib.download.download_testdata(
        model_url, "mobilenetv2-7.onnx", module="onnx"
    )
    return onnx.load(model_path)


@pytest.mark.skip("takes too long (~20min)")
@tvm.testing.requires_hexagon
def test_mobilenet_onnx(hexagon_session: Session):
    onnx_model = get_onnx_mobilenet()
    data_np = np.random.rand(1, 3, 224, 224).astype("float32")
    shape_dict = {"input": data_np.shape}
    relay_mod, _ = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target_hexagon)

    # Compile and run on Hexagon.
    ex = relax.vm.build(relax_mod, target)
    dev = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, dev)
    data = tvm.nd.array(data_np, dev)
    hexagon_res = vm_rt["main"](data)

    # Compile and run on LLVM for comparison.
    relax_mod = relay_translator.from_relay(relay_mod["main"], "llvm")
    ex = relax.vm.build(relax_mod, "llvm")
    dev = tvm.cpu()
    vm_rt = relax.VirtualMachine(ex, dev)
    data = tvm.nd.array(data_np, dev)
    llvm_res = vm_rt["main"](data)
    tvm.testing.assert_allclose(hexagon_res.numpy(), llvm_res.numpy(), rtol=1e-3)


@pytest.mark.skip("takes too long (~20min)")
@tvm.testing.requires_hexagon
def test_mobilenet(hexagon_session: Session):
    relay_mod, params = testing.mobilenet.get_workload(batch_size=1, dtype="float32")
    data_np = np.random.rand(1, 3, 224, 224).astype("float32")

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    # translate the relay mobilenet and bind params
    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)

    # Compile and run on Hexagon.
    ex = relax.vm.build(relax_mod, target)
    dev = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, dev)
    data = tvm.nd.array(data_np, dev)
    hexagon_res = vm_rt["main"](data)

    # Compile and run on LLVM for comparison.
    relax_mod = relay_translator.from_relay(relay_mod["main"], "llvm", params)
    ex = relax.vm.build(relax_mod, "llvm")
    dev = tvm.cpu()
    vm_rt = relax.VirtualMachine(ex, dev)
    data = tvm.nd.array(data_np, dev)
    llvm_res = vm_rt["main"](data)
    tvm.testing.assert_allclose(hexagon_res.numpy(), llvm_res.numpy(), rtol=1e-3)


@pytest.mark.skip("takes too long (~20min)")
@tvm.testing.requires_hexagon
def test_mobilenet_dyn(hexagon_session: Session):
    relay_mod, params = testing.mobilenet.get_workload(batch_size=relay.Any(), dtype="float32")
    data_np = np.random.rand(1, 3, 224, 224).astype("float32")

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    # translate the relay mobilenet and bind params
    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)

    # Compile and run on Hexagon.
    ex = relax.vm.build(relax_mod, target)
    dev = hexagon_session.device

    vm_mod = hexagon_session.get_executor_from_factory(ex)
    vm_rt = relax.VirtualMachine(vm_mod, dev)
    data = tvm.nd.array(data_np, dev)
    hexagon_res = vm_rt["main"](data)

    # Compile and run on Relay for comparison.
    dev = tvm.cpu()
    data = tvm.nd.array(data_np, dev)

    target = tvm.target.Target("llvm", host="llvm")
    vm_exec = relay.vm.compile(relay_mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    relay_res = vm_factory.invoke("main", data, **params)
    tvm.testing.assert_allclose(hexagon_res.numpy(), relay_res.numpy(), rtol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
