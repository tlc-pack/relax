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
# pylint: disable=missing-function-docstring,missing-module-docstring
import tvm
from tvm.script import tir as T
from tvm import te, tir
import numpy as np
import tvm.testing


def test_unique_name():
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16, 16), lambda x, y: A[x, y] * 2, name="main")
    C = te.compute((16, 16), lambda x, y: B[x, y] + 1, name="main")
    func = te.create_prim_func([A, C])
    s = tir.Schedule(func, debug_mask="all")
    assert isinstance(s.get_sref(s.get_block("main")), tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_block("main_1")), tir.schedule.StmtSRef)


def _check_workload(te_workload, tir_workload):
    func = te.create_prim_func(te_workload())
    tvm.ir.assert_structural_equal(func, tir_workload)
    # make sure that we can create schedule from the func
    s = tir.Schedule(func, debug_mask="all")
    assert s


def te_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [A, B, C]


@T.prim_func
def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i0, j0, k0 in T.grid(128, 128, 128):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            with T.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_matmul():
    _check_workload(te_matmul, tir_matmul)


def te_element_wise():
    A = te.placeholder((128, 128), name="A")
    B = te.compute((128, 128), lambda x, y: A[x, y] * 2, name="B")
    C = te.compute((128, 128), lambda x, y: B[x, y] + 1, name="C")
    return [A, C]


@T.prim_func
def tir_element_wise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))

    for i0, j0 in T.grid(128, 128):
        with T.block():
            i, j = T.axis.remap("SS", [i0, j0])
            B[i, j] = A[i, j] * 2.0
    for i0, j0 in T.grid(128, 128):
        with T.block():
            i, j = T.axis.remap("SS", [i0, j0])
            C[i, j] = B[i, j] + 1.0


def test_element_wise():
    _check_workload(te_element_wise, tir_element_wise)


def te_conv2d():
    batch = 16
    in_channel = 16
    out_channel = 32
    size = 14
    kernel = 3

    A = te.placeholder((batch, in_channel, size, size), name="A")
    W = te.placeholder((in_channel, kernel, kernel, out_channel), name="W")
    Apad = te.compute(
        (batch, in_channel, size + 2, size + 2),
        lambda nn, cc, yy, xx: tvm.tir.if_then_else(
            tvm.tir.all(yy >= 1, yy - 1 < size, xx >= 1, xx - 1 < size),
            A[nn, cc, yy - 1, xx - 1],
            0.0,
        ),
        name="Apad",
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    B = te.compute(
        (batch, out_channel, size, size),
        lambda nn, ff, yy, xx: te.sum(
            Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff], axis=[rc, ry, rx]
        ),
        name="B",
    )
    return [A, W, B]


@T.prim_func
def tir_conv2d(a: T.handle, w: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 14, 14])
    W = T.match_buffer(w, [16, 3, 3, 32])
    B = T.match_buffer(b, [16, 32, 14, 14])
    Apad = T.alloc_buffer([16, 16, 16, 16])

    for n, c, y, x in T.grid(16, 16, 16, 16):
        with T.block("Apad"):
            nn, cc, yy, xx = T.axis.remap("SSSS", [n, c, y, x])
            Apad[nn, cc, yy, xx] = T.if_then_else(
                yy >= 1 and yy - 1 < 14 and xx >= 1 and xx - 1 < 14,
                A[nn, cc, yy - 1, xx - 1],
                0.0,
                dtype="float32",
            )
    for n, f, y, x, kc, ky, kx in T.grid(16, 32, 14, 14, 16, 3, 3):
        with T.block("B"):
            nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [n, f, y, x, kc, ky, kx])
            with T.init():
                B[nn, ff, yy, xx] = 0.0
            B[nn, ff, yy, xx] += Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff]


def test_conv2d():
    _check_workload(te_conv2d, tir_conv2d)


def te_multi_output():
    n = te.var("n")
    m = te.var("m")
    A0 = te.placeholder((m, n), name="A0")
    A1 = te.placeholder((m, n), name="A1")
    B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name="B")
    return [A0, A1, B0, B1]


@T.prim_func
def tir_multi_output(a0: T.handle, a1: T.handle, b0: T.handle, b1: T.handle) -> None:
    m = T.var("int32")
    n = T.var("int32")
    A0 = T.match_buffer(a0, (m, n))
    A1 = T.match_buffer(a1, (m, n))
    B0 = T.match_buffer(b0, (m, n))
    B1 = T.match_buffer(b1, (m, n))

    for i0, i1 in T.grid(m, n):
        with T.block("B.v0"):
            i, j = T.axis.remap("SS", [i0, i1])
            B0[i, j] = A0[i, j] + 2.0
        with T.block("B.v1"):
            i, j = T.axis.remap("SS", [i0, i1])
            B1[i, j] = A1[i, j] * 3.0


def test_multi_output():
    _check_workload(te_multi_output, tir_multi_output)


def te_extern():
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.extern(
        (128, 128),
        [A, B],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], 0, 0
        ),
        name="C",
    )
    return [A, B, C]


@T.prim_func
def tir_extern(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    # body
    with T.block("C"):
        T.reads([A[0:128, 0:128], B[0:128, 0:128]])
        T.writes([C[0:128, 0:128]])
        T.evaluate(
            T.tvm_call_packed(
                "tvm.contrib.cblas.matmul",
                T.tvm_stack_make_array(
                    A.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    B.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    C.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
                    dtype="handle",
                ),
                0,
                0,
                dtype="int32",
            )
        )


def test_extern():
    _check_workload(te_extern, tir_extern)


def te_reordered_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [C, A, B]


@T.prim_func
def tir_reordered_matmul(c: T.handle, a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i0, j0, k0 in T.grid(128, 128, 128):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            with T.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_arg_order():
    _check_workload(te_reordered_matmul, tir_reordered_matmul)


def te_scan():
    m = te.var("m")
    n = te.var("n")
    X = te.placeholder((m, n), name="X")
    s_state = te.placeholder((m, n))
    s_init = te.compute((1, n), lambda _, i: X[0, i])
    s_update = te.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
    s_scan = tvm.te.scan(s_init, s_update, s_state, inputs=[X])
    return [X, s_scan]


def test_error_reporting():
    try:
        te.create_prim_func(te_scan())
        assert False
    except TypeError as e:
        error_message = str(e)
        assert error_message.find("Unsupported Operation: ScanOp.") != -1
        return
    assert False


def test_constant():
    M = 11
    A = te.placeholder((M,), name="A")
    B = te.compute(tuple(), lambda: 2, name="B")
    # Manually craft ProducerLoad because `B[]` is not allowed.
    C = te.compute(
        (M,), lambda x: A[x] + tvm.tir.expr.ProducerLoad(B, []), name="C", tag="broadcast"
    )

    func = te.create_prim_func([C, A])
    func = tvm.build(func)
    a_np = np.random.uniform(size=(M,)).astype(A.dtype)
    c = tvm.nd.array(np.zeros(M, dtype=C.dtype))
    x = func(c, tvm.nd.array(a_np))
    tvm.testing.assert_allclose(a_np + 2, c.numpy())


def test_data_dependent_access():
    A = te.placeholder((10,), name="A")
    B = te.placeholder((10,), name="B", dtype="int32")
    C = te.compute((10,), lambda i: A[B[i]])

    func = te.create_prim_func([C, A, B])
    func = tvm.build(func)

    a_np = np.random.uniform(size=(10,)).astype(A.dtype)
    b_np = np.arange(10, dtype=B.dtype)
    c = tvm.nd.array(np.zeros(10, dtype=C.dtype))
    func(c, tvm.nd.array(a_np), tvm.nd.array(b_np))
    tvm.testing.assert_allclose(a_np[b_np], c.numpy())


def test_loop_var_datatype():
    def test_helper(dtype):
        n = te.var("n", dtype)
        A = te.placeholder((n,), name="A")
        B = te.placeholder((n,), name="B", dtype="int32")
        C = te.compute((n,), lambda i: A[i] + B[i])

        func = te.create_prim_func([C, A, B])

        assert func.body.block.body.loop_var.dtype == dtype

        func = tvm.build(func)

        a_np = np.random.uniform(size=(10,)).astype(A.dtype)
        b_np = np.random.uniform(size=(10,)).astype(B.dtype)
        c = tvm.nd.array(np.zeros(10, dtype=C.dtype))
        func(c, tvm.nd.array(a_np), tvm.nd.array(b_np))
        tvm.testing.assert_allclose(a_np + b_np, c.numpy())

    test_helper("int32")
    test_helper("int64")


def test_unbound_var():
    n = tir.Var("n", "int32")
    A = te.placeholder((n + 1,), name="A")
    B = te.compute((n + 1,), lambda i: A[i], name="B")
    func = te.create_prim_func([A, B], [n])
    assert len(func.params) == 3
    assert func.params[2] == n

    func = tvm.build(func)

    a_np = np.random.uniform(size=(10,)).astype(A.dtype)
    b = tvm.nd.array(np.zeros(10, dtype=B.dtype))
    func(tvm.nd.array(a_np), b, 9)
    tvm.testing.assert_allclose(a_np, b.numpy())


def te_argmax():
    # x and y are the operands of reduction, both of them is a tuple of index
    # and value.
    def fcombine(x, y):
        lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    # our identity element also need to be a tuple, so `fidentity` accepts
    # two types as inputs.
    def fidentity(t0, t1):
        return tvm.tir.const(-1, t0), tvm.te.min_value(t1)

    argmax = te.comm_reducer(fcombine, fidentity, name="argmax")

    # describe the reduction computation
    m = te.var("m")
    n = te.var("n")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    val = te.placeholder((m, n), name="val", dtype="int32")
    k = te.reduce_axis((0, n), "k")
    T0, T1 = te.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="T")
    return [idx, val, T0, T1]


@T.prim_func
def tir_argmax(
    var_idx: T.handle, var_val: T.handle, var_T_v0: T.handle, var_T_v1: T.handle
) -> None:
    m = T.var("int32")
    n = T.var("int32")
    idx = T.match_buffer(var_idx, [m, n], dtype="int32")
    val = T.match_buffer(var_val, [m, n], dtype="int32")
    T_v0 = T.match_buffer(var_T_v0, [m], dtype="int32")
    T_v1 = T.match_buffer(var_T_v1, [m], dtype="int32")
    # body
    # with T.block("root")
    for i0, i1 in T.grid(m, n):
        with T.block("T.v0"):
            i, k = T.axis.remap("SR", [i0, i1])
            with T.init():
                T_v0[i] = -1
                T_v1[i] = -2147483648
            T_v0[i] = T.Select(T_v1[i] >= val[i, k], T_v0[i], idx[i, k])
            T_v1[i] = T.Select(T_v1[i] >= val[i, k], T_v1[i], val[i, k])


def test_argmax():
    _check_workload(te_argmax, tir_argmax)

    dtype = "int32"
    func = te.create_prim_func(te_argmax())
    assert len(func.params) == 4

    func = tvm.build(func)

    idx_np = np.arange(100, dtype=dtype).reshape((10, 10))
    val_np = np.random.permutation(100).reshape((10, 10)).astype(dtype)
    c = tvm.nd.array(np.zeros(10, dtype=dtype))  # argmax index
    d = tvm.nd.array(np.zeros(10, dtype=dtype))  # max value
    func(tvm.nd.array(idx_np), tvm.nd.array(val_np), c, d)

    c_expected = idx_np[np.arange(10), np.argmax(val_np, axis=1)]
    d_expected = np.amax(val_np, axis=1)

    tvm.testing.assert_allclose(c_expected, c.numpy())
    tvm.testing.assert_allclose(d_expected, d.numpy())


if __name__ == "__main__":
    test_unique_name()
    test_matmul()
    test_element_wise()
    test_conv2d()
    test_multi_output()
    test_extern()
    test_arg_order()
    test_error_reporting()
    test_constant()
    test_loop_var_datatype()
    test_unbound_var()
    test_argmax()
