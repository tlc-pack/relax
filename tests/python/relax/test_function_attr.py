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
from tvm.script import relax as R
from tvm import relax


def _check_equal(x, y):
    tvm.ir.assert_structural_equal(x, y)
    tvm.ir.assert_structural_equal(y, x)

    xhash = tvm.ir.structural_hash(x)
    yhash = tvm.ir.structural_hash(y)

    assert xhash == yhash


def _check_save_roundtrip(x):
    y = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, y)


@tvm.script.ir_module
class InputModule:
    @R.function
    def relax_add(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor:
        z1 = R.add(x, y)
        z2 = R.add(z1, z1)
        return z2

    @R.function
    def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor:
        lv0 = relax_add(x, y)
        return lv0


def annotate(mod, func_name, attrs):
    # Get func
    annot_func = mod[func_name]
    # Annotate a function
    for key, val in attrs.items():
        annot_func = annot_func.with_attr(key, val)
    mod[func_name] = annot_func
    return mod


def test_func_attr_setter():
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    mod = annotate(mod, "relax_add", {"Codegen": "test-codegen"})
    _check_save_roundtrip(mod)
    annot_func = mod["relax_add"]

    # Test annotation
    assert annot_func.attrs
    assert annot_func.attrs["Codegen"] == "test-codegen"


def test_func_attr_roundtrip_and_equality():
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    mod1 = annotate(mod, "relax_add", {"Codegen": "test-codegen"})
    mod2 = annotate(mod, "relax_add", {"Codegen": "test-codegen"})
    _check_save_roundtrip(mod1)
    _check_save_roundtrip(mod2)
    _check_equal(mod1, mod2)


def test_func_attr_setter_with_passes():
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    # Annotate
    mod = annotate(mod, "relax_add", {"Codegen": "test-codegen"})

    # Test with passes
    # Annotation should stay the same unless the pass needs to modify it

    # List of passes
    passes = [relax.transform.ToNonDataflow()]
    passes.append(relax.transform.CallTIRRewrite())
    passes.append(relax.transform.VMMemoryLower())
    passes.append(relax.transform.VMShapeLower())
    seq = tvm.transform.Sequential(passes)

    # Apply passes
    new_mod = seq(mod)
    print(mod.script())
    _check_save_roundtrip(new_mod)

    # Test annotation
    func = new_mod["relax_add"]
    assert func.attrs
    assert func.attrs["Codegen"] == "test-codegen"


def test_irmodule_attr_setter_with_passes():
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    # Annotate
    attr = relax.const(1, "float32")
    mod = mod.with_attr("test-attr", attr)
    mod_attr = mod.get_attrs()

    # Test with passes
    # Annotation should stay the same unless the pass needs to modify it

    # List of passes
    passes = [relax.transform.ToNonDataflow()]
    passes.append(relax.transform.CallTIRRewrite())
    passes.append(relax.transform.VMMemoryLower())
    passes.append(relax.transform.VMShapeLower())
    seq = tvm.transform.Sequential(passes)

    # Apply passes
    new_mod = seq(mod)
    _check_save_roundtrip(new_mod)

    # Check IRModule attrs is preserved after applying passes
    assert new_mod.get_attrs()["test-attr"] == attr
    assert new_mod.get_attrs() == mod_attr


if __name__ == "__main__":
    pytest.main([__file__])
