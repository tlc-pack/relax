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
import tvm
from tvm.script import relax as R
from tvm import relax


@tvm.script.ir_module
class InputModule:
    @R.function
    def relax_add(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
        z1 = relax.add(x, y)
        z2 = relax.add(z1, z1)
        return z2

    @R.function
    def main(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
        lv0 = relax_add(x, y)
        return lv0


def test_func_attr_setter():
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    # Annotate a function
    annot_mod = mod["relax_add"].with_attr("Codegen", "test-codegen")
    annot_mod = annot_mod.with_attr("global_symbol", "test-symbol")

    # Test annotation
    assert annot_mod.attrs
    assert annot_mod.attrs["Codegen"] == "test-codegen"
    assert annot_mod.attrs["global_symbol"] == "test-symbol"

    # Update ir module
    mod["relax_add"] = annot_mod

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

    # Test annotation
    func = new_mod["relax_add"]
    assert func.attrs
    assert func.attrs["Codegen"] == "test-codegen"
    assert func.attrs["global_symbol"] == "test-symbol"


if __name__ == "__main__":
    pytest.main([__file__])
