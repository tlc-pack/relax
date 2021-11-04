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
import os
import numpy as np
import tvm
from tvm.relay import Call
from tvm import relax
from tvm.runtime import container

import tvm.script
from tvm.script import tir as T, relax as R


@tvm.register_func("test.vm.move")
def move(src):
    return src


@tvm.register_func("test.vm.add")
def add(a, b):
    ret = a.asnumpy() + b.asnumpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.mul")
def mul(a, b):
    ret = a.asnumpy() * b.asnumpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.asnumpy())


@tvm.register_func("test.vm.tile")
def tile_packed(a, b):
    b[:] = tvm.nd.array(np.tile(a.asnumpy(), (1, 2)))


def test_vm_execute():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    b = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())


def test_vm_multiple_func():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    b = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    mul_res = vm["func1"](a, b)
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())
    np.testing.assert_allclose(mul_res.asnumpy(), a.asnumpy() * b.asnumpy())


def test_vm_serialize():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    arr = tvm.nd.array(
        np.random.rand(
            4,
        )
    )
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), arr], dst=ib.r(1))
        ib.emit_ret(ib.r(1))
    exec0 = ib.get()
    exec0.save_to_file("exec.tmp")
    exec1 = relax.load_exec_from_file("exec.tmp")
    assert exec0.astext() == exec1.astext()
    os.remove("exec.tmp")


def test_vm_constant_serialize():
    dtype = tvm.DataType("float32")
    shape = (4, 6)
    inp = tvm.nd.array(np.random.rand(4, 6).astype(np.float32))
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=1):
        ib.emit_call("vm.builtin.alloc_storage", args=[ib.vm_state(), (24,), ib.imm(1), dtype], dst=ib.r(1))
        ib.emit_call("vm.builtin.alloc_tensor", args=[ib.r(1), shape, ib.imm(0), dtype], dst=ib.r(2))
        ib.emit_call("test.vm.identity", args=[ib.r(0), ib.r(2)])
        ib.emit_ret(ib.r(2))
    exec0 = ib.get()
    exec0.save_to_file("exec.tmp")
    exec1 = relax.load_exec_from_file("exec.tmp")
    assert exec0.astext() == exec1.astext()
    vm = relax.VirtualMachine(exec0, tvm.cpu())
    res = vm["main"](inp)
    np.testing.assert_allclose(inp.asnumpy(), res.asnumpy())
    os.remove("exec.tmp")


def test_vm_checker():
    ib = relax.ExecBuilder()
    try:
        with ib.function("func0", num_inputs=2):
            ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(2)], dst=ib.r(2))
            ib.emit_ret(ib.r(2))
        ib.get()
    except ValueError as ex:
        assert True


def test_vm_formalize():
    ib0 = relax.ExecBuilder()
    ib1 = relax.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(100))
        ib0.emit_call("test.vm.mul", args=[ib0.r(1), ib0.r(100)], dst=ib0.r(50))
        ib0.emit_ret(ib0.r(50))
    with ib1.function("func0", num_inputs=2):
        ib1.emit_call("test.vm.add", args=[ib1.r(0), ib1.r(1)], dst=ib1.r(2))
        ib1.emit_call("test.vm.mul", args=[ib1.r(1), ib1.r(2)], dst=ib1.r(3))
        ib1.emit_ret(ib1.r(3))
    exec0 = ib0.get()
    exec1 = ib1.get()
    assert exec0.astext() == exec1.astext()


@tvm.register_func("test.vm.add_scalar")
def add_scalar(a, b):
    return a + b


@tvm.register_func("test.vm.get_device_id")
def get_device_id(device):
    return device.device_id


def test_vm_operand():
    ib0 = relax.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add_scalar", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(2))
        ib0.emit_ret(ib0.r(2))
    exec0 = ib0.get()
    vm = relax.VirtualMachine(exec0, tvm.cpu())
    res = vm["func0"](2, 3)
    assert res == 5

    ib1 = relax.ExecBuilder()
    with ib1.function("func1", num_inputs=1):
        ib1.emit_call("test.vm.get_device_id", args=[ib1.r(0)], dst=ib1.r(1))
        ib1.emit_ret(ib1.r(1))
    exec1 = ib1.get()
    vm = relax.VirtualMachine(exec1, tvm.cpu())
    res = vm["func1"](tvm.cpu(3))
    assert res == 3


def test_vm_shapeof():
    ib = relax.ExecBuilder()
    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    with ib.function("main", num_inputs=0):
        ib.emit_call("vm.builtin.shape_of", args=[arr], dst=ib.r(0))
        ib.emit_ret(ib.r(0))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    for i, s in enumerate(res):
        assert s == shape[i]


def test_vm_storage():
    dtype = tvm.DataType("float32")
    shape = (4, 6)
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=0):
        ib.emit_call("vm.builtin.alloc_storage", args=[ib.vm_state(), (24,), ib.imm(1), dtype], dst=ib.r(1))
        ib.emit_call("vm.builtin.alloc_tensor", args=[ib.r(1), shape, ib.imm(0), dtype], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape_tuple = container.ShapeTuple(shape)
    res = vm["main"]()
    assert res.device == tvm.cpu()
    assert res.shape == shape


def test_vm_compile_stage0():
    @tvm.script.ir_module
    class TestVMCompileStage0:
        @R.function
        def foo(x: Tensor[(3, 4), "float32"], y: Tensor[(3, 4), "float32"]):
            z = relax.call_packed("test.vm.identity", x, y)
            return y

    mod = TestVMCompileStage0
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = relax.vm.build(mod, target, target_host)
    inp1 = tvm.nd.array(np.random.rand(3,4).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3,4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)
    vm["foo"](inp1, inp2)
    np.testing.assert_allclose(inp2.asnumpy(), inp1.asnumpy())


def test_vm_compile_stage1():
    # FIXME(@altanh): see comment in test_parser.py
    src = """@tvm.script.ir_module
class TestVMCompileStage1:
    @T.prim_func
    def shape_func0(heap: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "shape_func0"})
        H = T.match_buffer(
            heap,
            [T.int64(4)],
            dtype="int64",
            elem_offset=T.int64(0),
            align=128,
            offset_factor=1,
        )
        # body
        T.store(
            H.data, T.int64(2), (T.load("int64", H.data, T.int64(0)) * T.int64(2)), True
        )
        T.store(
            H.data, T.int64(3), (T.load("int64", H.data, T.int64(1)) * T.int64(3)), True
        )

    @R.function
    def foo(x: Tensor[_, "float32"]) -> Shape:
        shape_heap: Tensor[(4,), "int64"] = relax.call_packed(
            "vm.builtin.alloc_shape_heap", (4,)
        )
        gv0 = relax.call_packed("vm.builtin.shape_of", x)
        gv1 = relax.call_packed("vm.builtin.store_shape", gv0, shape_heap, (0, 1))
        gv2 = shape_func0(shape_heap)
        gv3 = relax.call_packed("vm.builtin.load_shape", shape_heap, (2, 3))
        return gv3
"""

    mod = R.parser.from_source(src)
    code = R.parser.astext(mod)
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = relax.vm.build(mod, target, target_host)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage2():
    @tvm.script.ir_module
    class TestVMCompileStage2:
        @R.function
        def foo(x: Tensor[_, "float32"]) -> Shape:
            sh = relax.call_packed("vm.builtin.shape_of", x)
            relax.match_shape(sh, (n, m))
            return (n * 2, m * 3)

    mod = TestVMCompileStage2
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = relax.vm.build(mod, target, target_host)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage3():
    @tvm.script.ir_module
    class TestVMCompileStage3:
        @R.function
        def foo(x: Tensor[(32, 16), "float32"]) -> Tensor:
            with relax.dataflow():
                y = relax.call_dps((32, 16), "test.vm.identity", (x))
                relax.output(y)
            return y

    mod = TestVMCompileStage3
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = relax.vm.build(mod, target, target_host)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    np.testing.assert_allclose(inp.asnumpy(), res.asnumpy())


def test_vm_compile_e2e():
    @tvm.script.ir_module
    class TestVMCompileE2E:
        @R.function
        def foo(x: Tensor[_, "float32"]) -> Tensor:
            with relax.dataflow():
                sh = relax.call_packed("vm.builtin.shape_of", x)
                x0 = relax.match_shape(sh, (n, m))
                y = relax.call_dps((n, m * 2), "test.vm.tile", (x))
                relax.output(y)
            return y

    mod = TestVMCompileE2E

    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = relax.vm.build(mod, target, target_host)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    np.testing.assert_allclose(np.tile(inp.asnumpy(), (1, 2)), res.asnumpy())


if __name__ == "__main__":
    test_vm_execute()
    test_vm_multiple_func()
    test_vm_checker()
    test_vm_formalize()
    test_vm_operand()
    test_vm_serialize()
    test_vm_constant_serialize()
    test_vm_shapeof()
    test_vm_storage()
    test_vm_compile_stage0()
    test_vm_compile_stage1()
    test_vm_compile_stage2()
    test_vm_compile_stage3()
    test_vm_compile_e2e()
