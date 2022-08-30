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
from tvm import relax
from tvm.ir.base import assert_structural_equal

from tvm.runtime.object import Object

import tvm.script
from tvm.script import relax as R


def test_simple_assignments():
    @tvm.script.ir_module
    class TestChainAssignments:
        @R.function
        def main(x: Tensor):
            y = x
            z = y
            q = z
            p = q
            o = p
            return o

    # a little annoying to have these unused bindings around
    # but they can be eliminated in a separate pass
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor):
            y = x
            z = x
            q = x
            p = x
            o = x
            return x

    new_mod = relax.transform.CanonicalizeBindings()(TestChainAssignments)
    assert_structural_equal(new_mod, Expected)


def test_dataflow_block():
    @tvm.script.ir_module
    class TestDataflowAssignments:
        @R.function
        def main(x: Tensor):
            with R.dataflow():
                y = relax.const(1)
                z = y
                o = z
                p = o
                m = p
                n = m
                R.output(n)
            return n

    # a little annoying to have these unused bindings around
    # but they can be eliminated in a separate pass
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor):
            with R.dataflow():
                y = relax.const(1)
                z = y
                o = y
                p = y
                m = y
                # we can't get rid of n because it leaves the block
                n = y
                R.output(n)
            return n

    new_mod = relax.transform.CanonicalizeBindings()(TestDataflowAssignments)
    assert_structural_equal(new_mod, Expected)


def test_ops():
    @tvm.script.ir_module
    class TestOps:
        @R.function
        def main(x: Tensor, y: Tensor):
            w = y
            q = x
            z = relax.add(w, q)
            return relax.add(q, z)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor, y: Tensor):
            w = y
            q = x
            z = relax.add(y, x)
            return relax.add(x, z)

    new_mod = relax.transform.CanonicalizeBindings()(TestOps)
    assert_structural_equal(new_mod, Expected)


def test_casting():
    @tvm.script.ir_module
    class TestCasting:
        @R.function
        def main(x: Tensor) -> Object:
            y = x
            # z will be treated as object type even though it's a tensor
            z: Object = y
            return z

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor) -> Object:
            y = x
            # Cannot unify because the cast indicates user intent
            z: Object = x
            return z

    new_mod = relax.transform.CanonicalizeBindings()(TestCasting)
    assert_structural_equal(new_mod, Expected)


def test_match_shape():
    @tvm.script.ir_module
    class TestMatchShape:
        @R.function
        def main(x: Tensor):
            q = x
            z = R.match_shape(q, (m, n))
            w = z
            return w

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor):
            q = x
            # can't get rid of z because its shape_ is different from x's
            z = R.match_shape(x, (m, n))
            w = z
            return z

    new_mod = relax.transform.CanonicalizeBindings()(TestMatchShape)
    assert_structural_equal(new_mod, Expected)


def test_same_shape():
    @tvm.script.ir_module
    class TestSameShape:
        @R.function
        def main(x: Tensor((m, n), _)):
            y = x
            # trivial check
            z = R.match_shape(x, (m, n))
            w = z
            q = relax.add(w, y)
            return relax.add(q, w)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor((m, n), _)):
            y = x
            z = R.match_shape(x, (m, n))
            w = x
            q = relax.add(x, x)
            return relax.add(q, x)

    new_mod = relax.transform.CanonicalizeBindings()(TestSameShape)
    assert_structural_equal(new_mod, Expected)


def test_change_shape():
    @tvm.script.ir_module
    class TestSameShape:
        @R.function
        def main(x: Tensor((m, n), _)):
            y = x
            # not trivial: introduces new shape vars
            z = R.match_shape(x, (o, p))
            w = z
            q = relax.add(w, y)
            return relax.add(q, w)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor((m, n), _)):
            y = x
            z = R.match_shape(x, (o, p))
            w = z
            # the shape_ field on q will need to be updated
            q = relax.add(z, x)
            return relax.add(q, z)

    new_mod = relax.transform.CanonicalizeBindings()(TestSameShape)
    assert_structural_equal(new_mod, Expected)


if __name__ == "__main__":
    pytest.main([__file__])
