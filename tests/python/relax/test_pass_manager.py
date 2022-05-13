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
"""Unit tests for relax pass manager."""
from __future__ import annotations  # must import to defer parsing of annotations
import numpy as np
import pytest
import tvm
from tvm import relax, tir
from tvm.ir.base import assert_structural_equal
from tvm.relax import ExprMutator
from tvm.relax.expr import Call

import tvm.script
from tvm.script import tir as T, relax as R


def check_equal(mod1, mod2):
    mod1 = relax.transform.ToANF()(mod1)
    mod2 = relax.transform.ToANF()(mod2)
    assert_structural_equal(mod1, mod2)


def test_function_class_pass():
    @relax.transform.function_pass(opt_level=1)
    class TestReplaceFunc:
        """Simple test function to replace one argument to another."""

        def __init__(self, new_func):
            self.new_func = new_func

        def transform_function(self, func, mod, ctx):
            return self.new_func

    @tvm.script.ir_module
    class Before:
        @R.function
        def f1(x: Tensor((m, n), "float32")):
            return x

    @tvm.script.ir_module
    class Expected:
        @R.function
        def f2(x: Tensor((m, n), "float32")):
            gv0 = relax.add(x, x)
            return gv0

    fpass = TestReplaceFunc(Expected["f2"])
    assert fpass.info.opt_level == 1
    assert fpass.info.name == "TestReplaceFunc"
    After = fpass(Before)
    assert_structural_equal(After["f1"], Expected["f2"])


# Swap Multiply and Add Ops
class SwapMAVar(ExprMutator):
    def __init__(self):
        ExprMutator.__init__(self)

    def visit_call(self, call: Call):
        if call.op == tir.op.Op.get("relax.add"):
            new_op = tir.op.Op.get("relax.multiply")
        elif call.op == tir.op.Op.get("relax.multiply"):
            new_op = tir.op.Op.get("relax.add")
        else:
            new_op = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return Call(new_op, new_args, call.attrs, call.type_args, call.span)


def test_function_pass():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((m, n), "float32"), y: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, y)
                gv0 = relax.add(lv0, y)
                relax.output(gv0)
            gv1 = relax.multiply(x, y)
            gv2 = relax.add(gv1, y)
            return (gv0, gv1, gv2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor((m, n), "float32"), y: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.add(x, y)
                gv0 = relax.multiply(lv0, y)
                relax.output(gv0)
            gv1 = relax.add(x, y)
            gv2 = relax.multiply(gv1, y)
            return (gv0, gv1, gv2)

    pass_name = "function_pass_test"
    opt_level = 0

    # create FunctionPass with the function_pass decorator
    @relax.transform.function_pass(opt_level=opt_level, name=pass_name)
    def decorator_transform(expr, mod, ctx):
        return SwapMAVar().visit(expr)

    # check the transform info
    assert isinstance(decorator_transform, relax.transform.FunctionPass)
    assert decorator_transform.info.name == pass_name
    assert decorator_transform.info.opt_level == opt_level
    # run the transform
    After = decorator_transform(Before)
    check_equal(After, Expected)

    # create FunctionPass directly with the function_pass init call
    def direct_transform(expr, mod, ctx):
        return SwapMAVar().visit(expr)

    direct_transform = relax.transform.function_pass(direct_transform, opt_level=opt_level)
    assert isinstance(direct_transform, relax.transform.FunctionPass)
    assert direct_transform.info.name == "direct_transform"
    assert direct_transform.info.opt_level == opt_level
    # run the transform
    After = direct_transform(Before)
    check_equal(After, Expected)


def test_dataflowblock_class_pass():
    @relax.transform.dataflowblock_pass(opt_level=1)
    class TestReplaceBinding:
        """Simple test function to replace the first VarBinding to another."""

        def __init__(self):
            # create a new VarBinding
            type_anno = relax.DynTensorType(2, "float32")
            lv0 = relax.Var("lv1", (2, 2), type_anno)
            val = relax.const(np.random.rand(24, 56))
            self.new_binding = relax.VarBinding(lv0, val)

        def transform_dataflowblock(self, block, mod, ctx):
            bindings = block.bindings
            new_bindings = [self.new_binding, bindings[1]]
            new_block = relax.expr.DataflowBlock(new_bindings, block.span)
            return new_block

    @tvm.script.ir_module
    class Mod1:
        @R.function
        def f(x: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, x)
                gv0 = relax.add(x, x)
                relax.output(gv0)
            return gv0

    @tvm.script.ir_module
    class Mod2:
        @R.function
        def f(x: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.add(x, x)
                gv0 = relax.add(x, x)
                relax.output(gv0)
            return gv0

    block_pass = TestReplaceBinding()
    assert block_pass.info.opt_level == 1
    assert block_pass.info.name == "TestReplaceBinding"
    updated_mod1 = block_pass(Mod1)
    updated_mod2 = block_pass(Mod2)
    assert_structural_equal(updated_mod1["f"], updated_mod2["f"])


def test_dataflowblock_pass():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((m, n), "float32"), y: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, y)
                gv0 = relax.add(lv0, y)
                relax.output(gv0)
            gv1 = relax.multiply(x, y)
            gv2 = relax.add(gv1, y)
            return (gv0, gv1, gv2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor((m, n), "float32"), y: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.add(x, y)
                gv0 = relax.multiply(lv0, y)
                relax.output(gv0)
            gv1 = relax.multiply(x, y)
            gv2 = relax.add(gv1, y)
            return (gv0, gv1, gv2)

    pass_name = "dataflow_pass_test"
    opt_level = 0

    # create DataflowBlockPass with the dataflowblock_pass decorator
    @relax.transform.dataflowblock_pass(opt_level=opt_level, name=pass_name)
    def decorator_transform(expr, mod, ctx):
        return SwapMAVar().visit(expr)

    # check the transform info
    assert isinstance(decorator_transform, relax.transform.DataflowBlockPass)
    assert decorator_transform.info.name == pass_name
    assert decorator_transform.info.opt_level == opt_level
    # run the transform
    After = decorator_transform(Before)
    check_equal(After, Expected)

    # create DataflowBlockPass directly with the dataflowblock_pass init call
    def direct_transform(expr, mod, ctx):
        return SwapMAVar().visit(expr)

    direct_transform = relax.transform.dataflowblock_pass(direct_transform, opt_level=opt_level)
    assert isinstance(direct_transform, relax.transform.DataflowBlockPass)
    assert direct_transform.info.name == "direct_transform"
    assert direct_transform.info.opt_level == opt_level
    # run the transform
    After = direct_transform(Before)
    check_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__])
