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

from __future__ import annotations
import pytest

import re

import tvm
from tvm._ffi.base import TVMError
from tvm.relax.stmt_rewrite import DataflowBlockRewrite
from tvm.relax.analysis import name_to_binding
from tvm.relax.expr import DataflowVar, Var
from tvm.script import relax as R


@R.function
def identity(x: Tensor((32, 32), "float32")) -> Tensor:
    with R.dataflow():
        lv0 = x
        R.output(lv0)
    return lv0


def assert_immutability(rwt, original_dfb, original_root_fn):
    assert rwt.mutated_dfb() != original_dfb
    assert rwt.mutated_root_fn() != original_root_fn
    assert rwt.mutated_root_fn().body.blocks[0] != original_dfb
    assert rwt.mutated_root_fn().body.blocks[0] == rwt.mutated_dfb()


def check_ground_truth(lhs, rhs):
    def_unifier = re.compile(r"def [a-zA-Z0-9_]+")
    cls_unifier = re.compile(r"class [a-zA-Z0-9_]+")
    unifier = lambda s: def_unifier.sub("def fn", cls_unifier.sub("class Mod", s))

    lhs_script = unifier(lhs.script())
    rhs_script = unifier(rhs.script())
    assert lhs_script == rhs_script


def test_null_construct():
    dfb = identity.body.blocks[0]
    root_fn = identity

    DataflowBlockRewrite(dfb, root_fn)


def test_simple_add():
    dfb = identity.body.blocks[0]
    root_fn = identity

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(name="tmp", expr=identity.params[0], is_dfvar=True)

    assert_immutability(rwt, dfb, root_fn)

    # check "tmp" added
    assert "tmp" in name_to_binding(rwt.mutated_root_fn())

    @R.function
    def ground_truth(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            tmp: Tensor((32, 32), "float32") = x
            R.output(lv0)
        return lv0

    check_ground_truth(rwt.mutated_root_fn(), ground_truth)


def test_simple_auto_add_var():
    dfb = identity.body.blocks[0]
    root_fn = identity

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(identity.params[0], is_dfvar=False)

    assert isinstance(rwt.mutated_dfb().bindings[-1].var, Var)

    assert_immutability(rwt, dfb, root_fn)


def test_simple_auto_add_dfvar():
    dfb = identity.body.blocks[0]
    root_fn = identity

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(identity.params[0], is_dfvar=True)

    assert isinstance(rwt.mutated_dfb().bindings[-1].var, DataflowVar)

    # immutatbility
    assert_immutability(rwt, dfb, root_fn)


def test_simple_remove_unused():
    @R.function
    def identity_unused(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            unused = lv0 + R.const(1.0)
            R.output(lv0)
        return lv0

    dfb = identity_unused.body.blocks[0]
    root_fn = identity_unused

    n2binding = name_to_binding(identity_unused)

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_unused(n2binding["unused"][0].var)

    assert_immutability(rwt, dfb, root_fn)

    # check "unused" removed
    assert "unused" not in name_to_binding(rwt.mutated_root_fn())

    @R.function
    def ground_truth(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return lv0

    check_ground_truth(rwt.mutated_root_fn(), ground_truth)


def test_remove_unused_undef():
    dfb = identity.body.blocks[0]
    root_fn = identity

    with pytest.raises(TVMError):
        rwt = DataflowBlockRewrite(dfb, root_fn)
        rwt.remove_unused(Var("whatever"))

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_unused(Var("whatever"), allow_undef=True)

    assert identity == rwt.mutated_root_fn()


def test_simple_rm_all_unused():
    @R.function
    def identity_unused(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            unused0 = lv0 + R.const(1.0)
            unused1 = lv0 + R.const(1.0)
            R.output(lv0)
        return lv0

    dfb = identity_unused.body.blocks[0]
    root_fn = identity_unused

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_all_unused()

    @R.function
    def ground_truth(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return lv0

    check_ground_truth(rwt.mutated_root_fn(), ground_truth)


def test_chained_rm_all_unused():
    @R.function
    def identity_unused(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            unused0 = R.call_tir(my_sigmoid, (x,), (32, 32), dtype="float32")
            unused1 = R.call_tir(my_sigmoid, (unused0,), (32, 32), dtype="float32")
            R.output(lv0)
        return lv0

    dfb = identity_unused.body.blocks[0]
    root_fn = identity_unused

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_all_unused()

    @R.function
    def ground_truth(x: Tensor((32, 32), "float32")) -> Tensor:
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return lv0

    check_ground_truth(rwt.mutated_root_fn(), ground_truth)


def test_simple_replace_all_uses():
    @R.function
    def lv0to1(x: Tensor((32, 32), "float32")) -> Tensor((32, 32), "float32"):
        #   lv0 => lv1
        #  /   \
        # lv2  lv3
        #  \   /
        #   lv4
        with R.dataflow():
            lv0: Tensor((32, 32), "float32") = R.call_tir(my_relu, (x,), (32, 32), dtype="float32")
            lv1: Tensor((32, 32), "float32") = R.call_tir(
                my_sigmoid, (x,), (32, 32), dtype="float32"
            )
            lv2: Tensor((32, 32), "float32") = R.call_tir(
                my_add, (x, lv0), (32, 32), dtype="float32"
            )
            lv3: Tensor((32, 32), "float32") = R.call_tir(
                my_mul, (x, lv0), (32, 32), dtype="float32"
            )
            lv4: Tensor((32, 32), "float32") = R.call_tir(
                my_whatever, (lv2, lv3), (32, 32), dtype="float32"
            )
            R.output(lv4)
        return lv4

    root_fn = lv0to1
    dfb = root_fn.body.blocks[0]

    n2binding = name_to_binding(root_fn)

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.replace_all_uses(n2binding["lv0"][0].var, n2binding["lv1"][0].var)
    rwt.remove_unused(n2binding["lv0"][0].var)

    assert_immutability(rwt, dfb, root_fn)

    n2binding_after = name_to_binding(rwt.mutated_root_fn())
    assert "lv0" not in n2binding_after


def test_simple_module_update():
    @tvm.script.ir_module
    class Identity:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(name="tmp", expr=root_fn.params[0], is_dfvar=True)

    new_ir = rwt.mutate_irmodule(Identity)

    # immutatbility
    assert new_ir != Identity
    assert 2 == len(new_ir["main"].body.blocks[0].bindings)

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = x
                tmp: Tensor((32, 32), "float32") = x
                R.output(lv0)
            return lv0

    check_ground_truth(new_ir, GroundTruth)
