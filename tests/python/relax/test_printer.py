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
import pytest
import tvm
from tvm import relax as rx
from tvm import tir, relay
from tvm.ir import structural_equal, assert_structural_equal


def check_roundtrip(f_pre):
    f_post = rx.parser.fromtext(rx.parser.astext(f_pre))
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
        relax.match_shape(x.shape, (n, m))
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
            x2: Tensor[(n, m), _] = relax.match_shape(x, (n, m))
            y = add(x2, x2)
            z = multiply(y, x)
            relax.match_shape(z.shape, (n, m))
            w: Tensor[(n, m), _] = subtract(z, x)
            relax.output(y, w, x2)
        t: Tensor[(n, m), _] = divide(y, w)
        q: Tensor[(n, m), _] = add(t, x2)
        return q

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


def test_call_dps_extern():
    @rx.script
    def foo(x: Tensor):
        z = relax.call_dps((10,), "my_extern", (x,))
        return z

    check_roundtrip(foo)


def test_class_irmodule():
    @rx.script
    class my_module:
        def f(x: Tensor[(n, m), _]) -> Tensor:
            return g(x)

        def g(y: Tensor[(n, m), _]) -> Tensor:
            return y

    check_roundtrip(my_module)
