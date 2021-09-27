from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import relax as rx
from tvm import tir, relay
from tvm.ir import structural_equal, assert_structural_equal


def rx_func(func):
    return func.module[func.fn_name]


def check_roundtrip(fn):
    f_pre = rx_func(fn)
    f_post = rx.parser.fromtext(rx.parser.astext(f_pre))[fn.fn_name]
    assert_structural_equal(f_pre, f_post, map_free_vars=True)


def test_annotations():
    @rx.script
    def foo(x: Tensor[(32, m), "float32"], y: Tensor[(m, k), "float32"]) -> Tensor:
        z: Tensor[(32, k), "float32"] = nn.matmul(x, y, units=None)
        w: Tensor[_, _] = multiply(z, z)
        t = subtract(w, z)
        sh: Shape = t.shape
        return t

    check_roundtrip(foo)


def test_match_shape():
    @rx.script
    def foo(x: Tensor[_, "float32"]):
        relax.match_shape((n, m), x.shape)
        y: Tensor[(n, m), "float32"] = add(x, x)
        return x

    check_roundtrip(foo)



def test_if():
    @rx.script
    def foo(cond: Tensor[(), "bool"], x: Tensor[(1,), "float32"]):
        if cond:
            w = add(x, x)
            y = multiply(w, w)
        else:
            w = multiply(x, x)
            y = add(w, w)
        return y

    check_roundtrip(foo)


def test_tuple():
    @rx.script
    def foo(x: Tensor[_, _], y: Tensor[(32,), "float32"]):
        t: Tuple[Tensor[_, _], Tensor[(32,), "float32"]] = (x, y)
        return t

    check_roundtrip(foo)


def test_local_func():
    @rx.script
    def foo(x: Tensor[_, _]):
        def bar(y: Tensor[_, _]):
            return y

        y = bar(x)  # tests local function variable scoping
        return y

    check_roundtrip(foo)


def test_dataflow():
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow():
            # TODO: parse this
            # nonlocal y, w
            y = add(x, x)
            z = multiply(y, x)
            w = subtract(z, x)
            relax.output(y, w)
        t = divide(y, w)
        return t

    check_roundtrip(foo)


def test_dataflow_match_shape():
    @rx.script
    def foo(x: Tensor[_, _]):
        with relax.dataflow():
            y = add(x, x)
            z = multiply(y, x)
            relax.match_shape((n, m), z.shape)
            w: Tensor[(n, m), _] = subtract(z, x)
            relax.output(y, w)
        t: Tensor[(n, m), _] = divide(y, w)
        return t

    check_roundtrip(foo)


def test_inline_tir():
    @rx.script
    def foo(x: Tensor[(B, 128), "float32"], y: Tensor[(128, 128), "float32"]):
        @tvm.script.tir
        def my_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
            A = tir.match_buffer(a, [128, 128])
            B = tir.match_buffer(b, [128, 128])
            C = tir.match_buffer(c, [128, 128])

            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                with tir.init():
                    C[vi, vj] = tir.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        z = relax.call_dps((B, 128), my_matmul, (x, y))
        return z

    check_roundtrip(foo)


def test_call_packed():
    @rx.script
    def foo(x: Tensor[(3, 4), "float32"]):
        # test that we can intro dim vars
        z: Tensor[(n, m), "float32"] = relax.call_packed("contrib.my_matmul", (x, x), mp=False)
        return z

    check_roundtrip(foo)


def test_primexpr_arithmetic():
    @rx.script
    def foo(x: Tensor[(n, m), "float32"]):
        z: Tensor[(n * m,), "float32"] = relax.call_packed("my_flatten", (x,))
        sh: Shape = (n + m, n // m)
        return z

    check_roundtrip(foo)
