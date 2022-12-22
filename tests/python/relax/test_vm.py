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
import os
from typing import Any, Callable, List, Tuple

import sys
import tempfile
import numpy as np
import pytest
import tvm
from tvm.runtime.object import Object
import tvm.script
import tvm.testing
from tvm import relax, rpc, te, tir, topi, TVMError
from tvm.contrib import utils
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T


@tvm.register_func("test.vm.move")
def move(src):
    return src


@tvm.register_func("test.vm.add")
def add(a, b):
    ret = a.numpy() + b.numpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.mul")
def mul(a, b):
    ret = a.numpy() * b.numpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.equal_zero")
def equal_zero(a):
    ret = np.all((a.numpy() == 0))
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.subtract_one")
def subtract_one(a):
    ret = np.subtract(a.numpy(), 1)
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.numpy())


@tvm.register_func("test.vm.tile")
def tile_packed(a, b):
    b[:] = tvm.nd.array(np.tile(a.numpy(), (1, 2)))


def check_saved_func(vm: relax.VirtualMachine, func_name: str, *inputs: List[Any]) -> Object:
    # uses save_function to create a closure with the given inputs
    # and ensure the result is the same
    # (assumes the functions return tensors and that they're idempotent)
    saved_name = f"{func_name}_saved"
    vm.save_function(func_name, saved_name, *inputs)
    res1 = vm[func_name](*inputs)
    res2 = vm[saved_name]()
    tvm.testing.assert_allclose(res1.numpy(), res2.numpy(), rtol=1e-7, atol=1e-7)
    return res1


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
    add_res = check_saved_func(vm, "func0", a, b)
    tvm.testing.assert_allclose(add_res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)


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
    mul_res = check_saved_func(vm, "func1", a, b)
    add_res = check_saved_func(vm, "func0", a, b)
    tvm.testing.assert_allclose(add_res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(mul_res.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_exec_serialize_export_library():
    @tvm.script.ir_module
    class TestVMMove:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32")):
            z = R.call_packed("vm.builtin.copy", x, type_args=(R.Tensor(ndim=2, dtype="float32")))
            return z

    mod = TestVMMove
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    from tvm.contrib import utils

    temp_dir = utils.tempdir()
    path_exec = temp_dir.relpath("exec.so")
    ex.mod.export_library(path_exec)

    loaded_exec = relax.vm.Executable(tvm.runtime.load_module(path_exec))
    assert ex.as_text() == loaded_exec.as_text()


def test_vm_checker():
    ib = relax.ExecBuilder()
    with pytest.raises(TVMError):
        with ib.function("func0", num_inputs=2):
            ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(2)], dst=ib.r(2))
            ib.emit_ret(ib.r(2))
        ib.get()


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
    assert exec0.as_text() == exec1.as_text()


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
        ib.emit_call(
            "vm.builtin.alloc_storage", args=[ib.vm_state(), (24,), ib.imm(0), dtype], dst=ib.r(1)
        )
        ib.emit_call(
            "vm.builtin.alloc_tensor", args=[ib.r(1), ib.imm(0), shape, dtype], dst=ib.r(2)
        )
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    assert res.device == tvm.cpu()
    assert res.shape == shape


def test_vm_copy():
    @tvm.script.ir_module
    class TestVMMove:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32")):
            z = R.call_packed("vm.builtin.copy", x, type_args=(R.Tensor(ndim=2, dtype="float32")))
            return z

    mod = TestVMMove
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    inp = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = check_saved_func(vm, "foo", inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_goto():
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_goto(2)
        ib.emit_call("test.vm.mul", args=[ib.r(2), ib.r(1)], dst=ib.r(2))
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
    res = check_saved_func(vm, "main", a, b)
    tvm.testing.assert_allclose(res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_if():
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=3):
        ib.emit_if(ib.r(0), 3)
        ib.emit_call("test.vm.add", args=[ib.r(1), ib.r(2)], dst=ib.r(3))
        ib.emit_goto(2)
        ib.emit_call("test.vm.mul", args=[ib.r(1), ib.r(2)], dst=ib.r(3))
        ib.emit_ret(ib.r(3))
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
    res = vm["main"](tvm.nd.array(False), a, b)
    tvm.testing.assert_allclose(res.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["main"](tvm.nd.array(1), a, b)
    tvm.testing.assert_allclose(res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_if():
    @tvm.script.ir_module
    class TestVMCompileIf:
        @R.function
        def ife(cond: R.Tensor((), "bool"), x: R.Tensor((3, 4), "float32")) -> R.Tensor:
            if cond:
                w = R.call_packed("test.vm.add", x, x, type_args=(R.Tensor))
            else:
                w = R.call_packed("test.vm.mul", x, x, type_args=(R.Tensor))
            return w

    mod = TestVMCompileIf
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(3, 4))
    res = vm["ife"](tvm.nd.array(1), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](tvm.nd.array(True), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](tvm.nd.array(0), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](tvm.nd.array(False), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_stage0():
    @tvm.script.ir_module
    class TestVMCompileStage0:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_packed(
                "test.vm.identity", x, y, type_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return y

    mod = TestVMCompileStage0
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    inp1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm["foo"](inp1, inp2)
    tvm.testing.assert_allclose(inp2.numpy(), inp1.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_stage1():
    @tvm.script.ir_module
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
            H[2] = H[0] * T.int64(2)
            H[3] = H[1] * T.int64(3)

        @R.function
        def foo(x: R.Tensor(dtype="float32")):
            shape_heap: R.Tensor((4,), "int64") = R.call_packed(
                "vm.builtin.alloc_shape_heap", (4,), type_args=(R.Tensor(ndim=1, dtype="int64"))
            )
            gv0 = R.call_packed("vm.builtin.shape_of", x, type_args=R.Shape)
            gv1 = R.call_packed("vm.builtin.store_shape", gv0, shape_heap, (0, 1), type_args=R.Void)
            gv2 = shape_func0(shape_heap)
            gv3 = R.call_packed("vm.builtin.load_shape", shape_heap, (2, 3), type_args=R.Shape)
            return gv3

    mod = TestVMCompileStage1
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage2():
    @tvm.script.ir_module
    class TestVMCompileStage2:
        @R.function
        def foo(x: R.Tensor(dtype="float32")) -> R.Shape:
            n, m = T.var("int64"), T.var("int64")
            R.match_shape(x, (n, m))
            return (n * 2, m * 3)

    mod = TestVMCompileStage2
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage3():
    @tvm.script.ir_module
    class TestVMCompileStage3:
        @R.function
        def foo(x: R.Tensor((32, 16), "float32")) -> R.Tensor:
            with R.dataflow():
                y = R.call_tir("test.vm.identity", (x), (32, 16), dtype="float32")
                R.output(y)
            return y

    mod = TestVMCompileStage3
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_e2e():
    @tvm.script.ir_module
    class TestVMCompileE2E:
        @R.function
        def foo(x: R.Tensor(dtype="float32")) -> R.Tensor:
            with R.dataflow():
                n, m = T.var("int64"), T.var("int64")
                R.match_shape(x, (n, m))
                y = R.call_tir("test.vm.tile", (x), (n, m * 2), dtype="float32")
                R.output(y)
            return y

    mod = TestVMCompileE2E

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = check_saved_func(vm, "foo", inp)
    tvm.testing.assert_allclose(res.numpy(), np.tile(inp.numpy(), (1, 2)), rtol=1e-7, atol=1e-7)


def test_vm_compile_e2e_func_param_with_shape():
    @tvm.script.ir_module
    class TestVMCompileE2E2:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def func(
            x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")
        ) -> R.Tensor:
            m, k = T.var("int64"), T.var("int64")
            gv0 = R.call_tir(tir_matmul, (x, w), (m, k), dtype="float32")
            return gv0

    mod = TestVMCompileE2E2

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    data = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    res = check_saved_func(vm, "func", data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_vm_emit_te_extern():
    if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
        print("skip because extern function is not available")
        return
    bb = relax.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor([n, m], "float32"))
    y = relax.Var("y", R.Tensor([m, n], "float32"))

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    data = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    res = check_saved_func(vm, "rx_cblas_matmul", data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_vm_emit_te_concat():
    # concatenate of two vectors of size (n,) and (m,)
    bb = relax.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))
    y = relax.Var("y", R.Tensor([m], "float32"))

    def te_func(A, B):
        C = te.compute((n + m), lambda i: tvm.tir.if_then_else(i < n, A[i], B[i - n]))
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    inp2 = tvm.nd.array(
        np.random.rand(
            2,
        ).astype(np.float32)
    )
    res = check_saved_func(vm, "rx_func", inp, inp2)
    tvm.testing.assert_allclose(
        res.numpy(), np.append(inp.numpy(), inp2.numpy()), rtol=1e-7, atol=1e-7
    )


def test_vm_emit_te_dtype_change():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))

    # convert a tensor with dtype of float32 to int16
    def te_func(A):
        B = te.compute((n,), lambda i: A[i].astype("int16"))
        return B

    with bb.function("rx_func", [x]):
        y = bb.emit_te(te_func, x)
        bb.emit_func_output(y)

    mod = bb.get()

    new_mod = relax.transform.CallTIRRewrite()(mod)
    assert new_mod["rx_func"].body.blocks[0].bindings[0].value.attrs.dtype == "int16"

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    res = check_saved_func(vm, "rx_func", inp)
    np.testing.assert_allclose(res.numpy(), inp.numpy().astype("int16"))


def test_vm_emit_te_floor_symbolic_shape():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))

    def te_func(A):
        C = te.compute((tir.floordiv(n, 2),), lambda i: A[i] + 1)
        return C

    with bb.function("rx_func", [x]):
        x1 = bb.emit_te(te_func, x)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape = (9,)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = check_saved_func(vm, "rx_func", inp)

    def expected_output():
        output_shape = (shape[0] // 2,)
        return inp.numpy()[: output_shape[0]] + 1

    tvm.testing.assert_allclose(res.numpy(), expected_output(), rtol=1e-7, atol=1e-7)


def test_vm_emit_te_constant_param_cpu():
    x_np = np.random.rand(2, 2).astype("float32")
    c_np = np.random.rand(2, 2).astype("float32")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 2), "float32"))
    c = relax.const(c_np, "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, c)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    mod = bb.get()
    exec = relax.vm.build(mod, "llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)

    add_res = check_saved_func(vm, "main", tvm.nd.array(x_np, dev))
    tvm.testing.assert_allclose(add_res.numpy(), x_np + c_np, rtol=1e-7, atol=1e-7)


@tvm.testing.requires_gpu
def test_vm_emit_te_constant_param_gpu():
    x_np = np.random.rand(2, 2).astype("float32")
    c_np = np.random.rand(2, 2).astype("float32")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 2), "float32"))
    c = relax.const(c_np, "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, c)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    mod = bb.get()
    sch = tvm.tir.Schedule(mod, debug_mask="all")
    loops = sch.get_loops(sch.get_block(name="T_add", func_name="add"))
    sch.bind(loops[0], "threadIdx.x")

    exec = relax.vm.build(sch.mod, "cuda")
    dev = tvm.cuda()
    vm = relax.VirtualMachine(exec, dev)

    add_res = check_saved_func(vm, "main", tvm.nd.array(x_np, dev))
    tvm.testing.assert_allclose(add_res.numpy(), x_np + c_np, rtol=1e-7, atol=1e-7)


def test_vm_relax_symbolic_shape():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))
    y = relax.Var("y", R.Tensor([(n // 2) + 1], "float32"))

    def te_func(A, B):
        C = te.compute((n,), lambda i: A[i] + B[i // 2])
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape1 = (5,)
    shape2 = (3,)
    inp = tvm.nd.array(np.random.rand(*shape1).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(*shape2).astype(np.float32))
    res = check_saved_func(vm, "rx_func", inp, inp2)

    def expected_output():
        return inp.numpy() + np.repeat(inp2.numpy(), 2)[:5]

    tvm.testing.assert_allclose(res.numpy(), expected_output(), rtol=1e-7, atol=1e-7)


def test_vm_relax_dyn_tir_shape():
    # case where TIR variables are unbound in generated PrimFunc
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")

    def te_func(A):
        C = te.compute((n + 1), lambda i: A[i])
        return C

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n + 1,), dtype="float32", name="y")

        x1 = bb.emit_te(te_func, y)
        bb.emit_func_output(x1, params=[x, y])

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    ex.mod.export_library("exec.so")
    exec1 = relax.vm.Executable(tvm.runtime.load_module("exec.so"))
    os.remove("exec.so")
    assert ex.as_text() == exec1.as_text()

    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(2).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3).astype(np.float32))

    res = check_saved_func(vm, "rx_func", inp, inp2)

    tvm.testing.assert_allclose(res.numpy(), inp2.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_tuple():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n,), dtype="float32", name="y")
        tup = relax.Tuple([x, y])
        item = tup[0]
        bb.emit_func_output([tup, item], params=[x, y])

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape = (5, 5)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    (res1, res2), res3 = vm["rx_func"](inp, inp2)

    tvm.testing.assert_allclose(res1.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(res2.numpy(), inp2.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(res3.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_tuplegetitem():
    @tvm.script.ir_module
    class TestVMTupleGetItem:
        @R.function
        def tuple_get_item(
            x: R.Tensor(ndim=2, dtype="float32"),
            y: R.Tensor(ndim=2, dtype="float32"),
        ):
            t = (x, y)
            a = t[0]
            b = t[1]
            c = R.call_packed("test.vm.add", a, b, type_args=(R.Tensor(ndim=2, dtype="float32")))
            return c

    mod = TestVMTupleGetItem
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(2, 3))
    y_inp = tvm.nd.array(np.random.rand(2, 3))
    res = check_saved_func(vm, "tuple_get_item", x_inp, y_inp)
    tvm.testing.assert_allclose(res.numpy(), x_inp.numpy() + y_inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_print_const():
    @tvm.script.ir_module
    class PrintConst:
        @R.function
        def main():
            x = R.const([1, 2])
            y = R.print(x)
            return x

    try:
        stdout = sys.stdout
        with tempfile.TemporaryFile(mode="w+") as test_out:
            sys.stdout = test_out
            mod = PrintConst
            target = tvm.target.Target("llvm", host="llvm")
            ex = relax.vm.build(mod, target)
            vm = relax.VirtualMachine(ex, tvm.cpu())
            res = vm["main"]()
            test_out.seek(0)
            printed_text = str(test_out.read())
            expected = "[1 2]\n"
            assert printed_text == expected
            tvm.testing.assert_allclose(res.numpy(), np.array([1, 2]))
    finally:
        sys.stdout = stdout


def test_vm_return_const_tuple():
    @tvm.script.ir_module
    class ReturnConstTuple:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")):
            y = R.const([1, 2])
            z = (y, R.const([3, 4]), x)
            return z

    mod = ReturnConstTuple
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(2, 3))
    res0, res1, res2 = vm["main"](inp)
    tvm.testing.assert_allclose(res0.numpy(), np.array([1, 2]))
    tvm.testing.assert_allclose(res1.numpy(), np.array([3, 4]))
    tvm.testing.assert_allclose(res2.numpy(), inp.numpy())


def test_vm_const_as_call_arg():
    @tvm.script.ir_module
    class TestVMConstAsCallArg:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")):
            a = R.call_packed(
                "test.vm.add",
                relax.const([1, 2]),
                relax.const([3, 4]),
                type_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            b = R.call_packed(
                "test.vm.add",
                a,
                x,
                type_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            return b

    mod = TestVMConstAsCallArg
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(1, 2))
    res = vm["main"](inp)
    tvm.testing.assert_allclose(res.numpy(), np.array([4, 6]) + inp.numpy())


def test_vm_if_cond_const():
    @tvm.script.ir_module
    class TestVMIfCondConst:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")) -> R.Tensor(ndim=2, dtype="float32"):
            if relax.const(True, dtype="bool"):
                ret = x
            else:
                ret = x
            return ret

    mod = TestVMIfCondConst
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(3, 4))
    res = vm["main"](inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy())


def test_sub_func_call():
    @tvm.script.ir_module
    class TestVMSubFunction:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def relax_matmul_tir(
            x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
        ) -> R.Tensor((32, 32), dtype="float32"):
            with R.dataflow():
                gv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
                R.output(gv0)
            return gv0

        @R.function
        def relax_matmul_packed(
            x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
        ) -> R.Object:
            gv0 = R.call_packed("test.vm.mul", x, w, type_args=(R.Tensor(ndim=2, dtype="float32")))
            return gv0

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Object:
            gv0 = relax_matmul_tir(x, w)
            gv1 = relax_matmul_packed(gv0, gv0)
            return gv1

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(TestVMSubFunction, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(32, 32).astype(np.float32))
    y_inp = tvm.nd.array(np.random.rand(32, 32).astype(np.float32))
    res = check_saved_func(vm, "main", x_inp, y_inp)
    product = np.dot(x_inp.numpy(), y_inp.numpy())
    expected = product * product
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_recursion():
    @tvm.script.ir_module
    class TestVMRecursion:
        @R.function
        def recursion(n: R.Tensor((1,), "float32")) -> R.Tensor:
            cond = R.call_packed(
                "test.vm.equal_zero", n, type_args=(R.Tensor(ndim=1, dtype="float32"))
            )
            if cond:
                res = R.const(1.0)
            else:
                gv0 = R.call_packed(
                    "test.vm.subtract_one", n, type_args=(R.Tensor(ndim=1, dtype="float32"))
                )
                tmp = recursion(gv0)
                res = R.call_packed(
                    "test.vm.add", tmp, tmp, type_args=(R.Tensor(ndim=1, dtype="float32"))
                )
            return res

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(TestVMRecursion, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    inp = np.empty(1)
    recursion_runs = np.random.randint(1, 10)
    inp.fill(recursion_runs)
    inp = tvm.nd.array(inp)
    res = check_saved_func(vm, "recursion", inp)
    tvm.testing.assert_allclose(res.numpy(), np.power(2.0, recursion_runs), rtol=1e-7, atol=1e-7)


def test_vm_closure():
    @tvm.script.ir_module
    class TestClosure:
        @R.function
        def lifted_func_1(x: R.Tensor((2, 3), "float32"), env: R.Tensor((2, 3), "float32")):
            return R.call_packed("test.vm.add", x, env, type_args=(R.Tensor))

        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 3), "float32"),
        ):
            clo = R.make_closure(lifted_func_1, (x,))
            res = R.invoke_closure(clo, (y,), type_args=(R.Tensor))
            return res

    mod = TestClosure
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(2, 3))
    y_inp = tvm.nd.array([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]])
    res = check_saved_func(vm, "main", x_inp, y_inp)
    tvm.testing.assert_allclose(res.numpy(), x_inp.numpy() + y_inp.numpy())


def test_vm_invoke_closure():
    ib = relax.ExecBuilder()
    with ib.function("lifted_func_1", num_inputs=4):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(4))
        ib.emit_call("test.vm.add", args=[ib.r(2), ib.r(4)], dst=ib.r(5))
        ib.emit_call("test.vm.add", args=[ib.r(3), ib.r(5)], dst=ib.r(6))
        ib.emit_ret(ib.r(6))
    with ib.function("main", num_inputs=2):
        x = ib.emit_constant("lifted_func_1")
        ib.emit_call("vm.builtin.alloc_closure", args=[ib.c(x), ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))

    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm.cpu())
    w_inp = tvm.nd.array(np.random.rand(2, 3))
    x_inp = tvm.nd.array(np.random.rand(2, 3))
    y_inp = tvm.nd.array([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]])
    z_inp = tvm.nd.array(np.random.rand(2, 3))
    clo = vm["main"](w_inp, x_inp)
    res = vm.invoke_closure(clo, (y_inp, z_inp))
    tvm.testing.assert_allclose(
        res.numpy(), w_inp.numpy() + x_inp.numpy() + y_inp.numpy() + z_inp.numpy()
    )


def test_time_evaluator():
    @tvm.script.ir_module
    class TestTimeEvaluator:
        @R.function
        def main(x: R.Tensor((1,), "float32"), y: R.Tensor((1,), "float32")):
            return R.call_packed("test.vm.add", x, y, type_args=(R.Tensor(ndim=1, dtype="float32")))

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(TestTimeEvaluator, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x = tvm.nd.array(np.random.rand(1))
    y = tvm.nd.array(np.random.rand(1))

    # ensure we can use time_evaluator with the stateful API
    vm.set_input("main", x, y)
    timing_res = vm.time_evaluator("invoke_stateful", tvm.cpu())("main")
    # just checking that it has some results at all
    assert timing_res.results

    # ensure we can use it with a closure
    vm.save_function("main", "saved_main", x, y)
    timing_res = vm.time_evaluator("saved_main", tvm.cpu())()
    assert timing_res.results


@tvm.script.ir_module
class TestVMSetInput:
    @T.prim_func
    def test_vm_mul(x: T.handle, y: T.handle, z: T.handle):
        T.func_attr({"global_symbol": "test_vm_mul"})
        m = T.var("int32")
        n = T.var("int32")
        A = T.match_buffer(x, (m, n))
        B = T.match_buffer(y, (m, n))
        C = T.match_buffer(z, (m, n))

        for i, j in T.grid(m, n):
            with T.block("mul"):
                vi = T.axis.spatial(m, i)
                vj = T.axis.spatial(n, j)
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    # test returning a tuple
    @R.function
    def test_vm_tuple(
        x: R.Tensor((), "int32")
    ) -> R.Tuple(R.Tensor((), "int32"), R.Tensor((), "int32")):
        return (x, x)

    # nested tuple too
    @R.function
    def test_vm_nested_tuple(
        x: R.Tensor((), "int32")
    ) -> R.Tuple(
        R.Tuple(
            R.Tensor((), "int32"),
            R.Tuple(
                R.Tensor((), "int32"),
            ),
        ),
        R.Tensor((), "int32"),
    ):
        return ((x, (x,)), x)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        gv0 = R.call_tir("test_vm_mul", (x, w), (32, 32), dtype="float32")
        return gv0


def set_input_trial(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    vm.invoke_stateful("main")
    res0 = vm.get_outputs("main")

    data_dict = {"x": a, "w": b}
    vm.set_input("main", **data_dict)
    vm.invoke_stateful("main")
    res1 = vm.get_outputs("main")
    tvm.testing.assert_allclose(res0.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(res0.numpy(), res1.numpy(), rtol=1e-7, atol=1e-7)

    # bug! If you don't bind the NDArray to a var, the memory will get corrupted.
    # Possibly due to object lifecycles and other FFI issues
    a = tvm.nd.array(2, device)
    vm.set_input("test_vm_tuple", a)
    vm.invoke_stateful("test_vm_tuple")
    res2 = vm.get_outputs("test_vm_tuple")
    # the results are NDArrays wrapped around scalars,
    # so we have to get the scalar out of the NDArray
    assert tuple(map(lambda a: int(a.numpy()), res2)) == (2, 2)

    b = tvm.nd.array(1, device)
    vm.set_input("test_vm_nested_tuple", b)
    vm.invoke_stateful("test_vm_nested_tuple")
    res3 = vm.get_outputs("test_vm_nested_tuple")
    assert len(res3) == 2 and len(res3[0]) == 2 and len(res3[0][1]) == 1
    result_cast = ((int(res3[0][0].numpy()), (int(res3[0][1][0].numpy()),)), int(res3[1].numpy()))
    assert result_cast == ((1, (1,)), 1)


def set_input_attempt_stateless(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # this should fail: once you set inputs, you cannot run statelessly
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    # must use invoke stateful!
    vm["main"]()


def set_input_attempt_invoke(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # this should fail: if the function needs inputs, you can't invoke directly
    vm.invoke_stateful("main")


def set_input_attempt_get(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # this should fail: you can't get outputs without invoking the function first
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    _ = vm.get_outputs("main")


def make_vm(mod) -> Tuple[relax.VirtualMachine, tvm.runtime.Device]:
    """Returns a local VM for the given mod and the device"""
    target = tvm.target.Target("llvm", host="llvm")
    exec = relax.vm.build(TestVMSetInput, target)
    exec.mod.export_library("exec.so")
    exec_loaded = relax.vm.Executable(tvm.runtime.load_module("exec.so"))
    os.remove("exec.so")
    device = tvm.cpu()
    return relax.VirtualMachine(exec_loaded, device), device


def run_on_rpc(
    mod: tvm.IRModule, trial_func: Callable[[relax.VirtualMachine, tvm.runtime.Device], None]
):
    """
    Sets up a VM over localhost using the given mod and runs the given trial function.
    The trial function should take a VM and a device
    """
    target = tvm.target.Target("llvm", host="llvm")
    exec = relax.vm.build(mod, target)
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    exec.mod.export_library(path)

    # Use local rpc server for testing.
    # Server must use popen so it doesn't inherit the current process state. It
    # will crash otherwise.
    # Adapted from relay/test_vm.py
    def check_remote(server):
        remote = rpc.connect(server.host, server.port, session_timeout=10)

        # Upload the serialized Executable.
        remote.upload(path)
        # Get a handle to remote Executable.
        rexec = remote.load_module("vm_library.so")

        device = remote.cpu()
        # Build a VM out of the executable and context.
        vm = relax.vm.VirtualMachine(exec=rexec, device=device)
        trial_func(vm, device)

    check_remote(rpc.Server("127.0.0.1"))


def test_set_input():
    set_input_trial(*make_vm(TestVMSetInput))


def test_set_input_rpc():
    run_on_rpc(TestVMSetInput, set_input_trial)


def save_function_kwargs_trial(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # just checking that we can use kwargs for the args when saving a function
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.save_function("main", "saved_main", x=a, w=b)
    res0 = vm["saved_main"]()
    tvm.testing.assert_allclose(res0.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)


def test_save_function_kwargs():
    save_function_kwargs_trial(*make_vm(TestVMSetInput))


def test_save_function_kwargs_rpc():
    run_on_rpc(TestVMSetInput, save_function_kwargs_trial)


def save_function_time_evaluator_trial(
    vm: relax.VirtualMachine, device: tvm.runtime.Device
) -> None:
    # just checking that the saved function can be called in the time evaluator
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.save_function("main", "saved_main", a, b)
    vm.time_evaluator("saved_main", device)()


def test_save_function_time_evaluator():
    save_function_time_evaluator_trial(*make_vm(TestVMSetInput))


def test_save_function_time_evaluator():
    run_on_rpc(TestVMSetInput, save_function_time_evaluator_trial)


# if you set an input, you should not be able to call statelessly
@pytest.mark.xfail()
def test_set_input_stateless_failure():
    set_input_attempt_stateless(*make_vm(TestVMSetInput))


@pytest.mark.xfail()
def test_set_input_stateless_failure_rpc():
    run_on_rpc(TestVMSetInput, set_input_attempt_stateless)


@pytest.mark.xfail()
def test_set_input_invoke_failure():
    set_input_attempt_invoke(*make_vm(TestVMSetInput))


@pytest.mark.xfail()
def test_set_input_invoke_failure_rpc():
    run_on_rpc(TestVMSetInput, set_input_attempt_invoke)


@pytest.mark.xfail()
def test_set_input_get_failure():
    set_input_attempt_get(*make_vm(TestVMSetInput))


@pytest.mark.xfail()
def test_set_input_get_failure_rpc():
    run_on_rpc(TestVMSetInput, set_input_attempt_get)


if __name__ == "__main__":
    tvm.testing.main()
