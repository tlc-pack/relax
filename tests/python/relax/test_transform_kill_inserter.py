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
from tvm import relax
from tvm import tir
from tvm.ir import assert_structural_equal
from tvm.script import tir as T, relax as R


def _check(mod_before, mod_expected):
    passes = [
        relax.transform.ToNonDataflow(),
        relax.transform.CallTIRRewrite(),
        relax.transform.InsertMemoryKills(),
    ]
    seq = tvm.transform.Sequential(passes)

    mod_after = seq(mod_before)
    assert_structural_equal(mod_after, mod_expected)


def test_linear():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def func_0(
                i: Tensor((), "float32"), s: Tensor((2, 3), "float32")) -> Tensor((2, 3), "float32"):
            tensor_1 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
            tensor_2 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
            new_i: Tensor((), "float32") = relax.add(i, tensor_2)
            new_s: Tensor((2, 3), "float32") = relax.add(s, tensor_1)
            output: Tensor((2, 3), "float32") = relax.add(new_i, new_s)
            return output

    @tvm.script.ir_module
    class Expected:
        @R.function
        def func_0(i: Tensor((), "float32"), s: Tensor((2, 3), "float32")) -> Tensor(None, "float32", ndim = 2):
            # block 0
            tensor_1: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
            tensor_2: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
            new_i: Tensor((), "float32") = relax.add(i, tensor_2)
            kill_tensor: Tuple() = relax.memory.kill_tensor(tensor_2)
            new_s: Tensor((2, 3), "float32") = relax.add(s, tensor_1)
            kill_tensor1: Tuple() = relax.memory.kill_tensor(tensor_1)
            output: Tensor((2, 3), "float32") = relax.add(new_i, new_s)
            kill_tensor2: Tuple() = relax.memory.kill_tensor(new_i)
            kill_tensor3: Tuple() = relax.memory.kill_tensor(new_s)
            return output
    # fmt: on

    before_mod = Before
    expected_mod = Expected
    _check(before_mod, expected_mod)


def test_simple_if():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def func_0(
                i: Tensor((), "float32"), s: Tensor((2, 3), "float32"), x: Tensor((2, 3), "float32")
        ) -> Tensor((2, 3), "float32"):
            tensor_1 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
            cond: Tensor((), "bool") = relax.call_packed(
                "test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool"))
            )
            tensor_3 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
            tensor_5: Tensor((2, 3), _) = relax.match_shape(relax.add(tensor_3, tensor_1), (2, 3))
            if cond:
                tensor_2 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
                new_i: Tensor((), "float32") = relax.add(i, tensor_2)
                new_s1: Tensor((2, 3), "float32") = relax.add(s, x)
                res: Tensor((2, 3), "float32") = relax.add(new_i, new_s1)
                output = res
            else:
                new_s: Tensor((2, 3), "float32") = relax.add(s, x)
                res: Tensor((2, 3), "float32") = relax.add(tensor_1, new_s)
                output = res
            return output

    @tvm.script.ir_module
    class Expected:
        @R.function
        def func_0(i: Tensor((), "float32"), s: Tensor((2, 3), "float32"), x: Tensor((2, 3), "float32")) -> Tensor(None, "float32", ndim = 2):
            # block 0
            tensor_1: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
            cond: Tensor((), "bool") = R.call_packed("test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool")))
            tensor_3: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
            tensor_5: Tensor((2, 3), "float32") = R.match_shape(relax.add(tensor_3, tensor_1), (2, 3))
            # <--- tensor_5 is never killed because it is not used
            kill_tensor: Tuple() = relax.memory.kill_tensor(tensor_3)
            if cond:
                # <--- tensor_1 should be killed here
                # block 0
                tensor_2: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
                new_i: Tensor((), "float32") = relax.add(i, tensor_2)
                kill_tensor1: Tuple() = relax.memory.kill_tensor(tensor_2)
                new_s1: Tensor((2, 3), "float32") = relax.add(s, x)
                res: Tensor((2, 3), "float32") = relax.add(new_i, new_s1)
                kill_tensor2: Tuple() = relax.memory.kill_tensor(new_i)
                kill_tensor3: Tuple() = relax.memory.kill_tensor(new_s1)
                output = res
            else:
                # block 0
                new_s: Tensor((2, 3), "float32") = relax.add(s, x)
                res: Tensor((2, 3), "float32") = relax.add(tensor_1, new_s)
                kill_tensor4: Tuple() = relax.memory.kill_tensor(tensor_1)
                kill_tensor5: Tuple() = relax.memory.kill_tensor(new_s)
                output = res
            return output
    # fmt: on

    before_mod = Before
    expected_mod = Expected
    _check(before_mod, expected_mod)


def test_nested_if():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def func_0(
                i: Tensor((), "float32"), s: Tensor((2, 3), "float32"), x: Tensor((2, 3), "float32")
        ) -> Tensor((2, 3), "float32"):
            cond: Tensor((), "bool") = relax.call_packed(
                "test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool"))
            )
            if cond:
                another_cond: Tensor((), "bool") = relax.call_packed(
                    "test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool"))
                )
                if another_cond:
                    tensor_3 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
                    new_s1 = relax.add(s, tensor_3)
                    new_s2: Tensor((2, 3), "float32") = relax.add(s, new_s1)
                else:
                    tensor_4 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
                    tensor_5 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
                    new_s1 = relax.add(tensor_5, tensor_4)
                    new_s2 = new_s1
                output = new_s2
            else:
                tensor_1 = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0)
                new_s3 = relax.add(tensor_1, x)
                output = new_s3
            return output

    @tvm.script.ir_module
    class Expected:
        @R.function
        def func_0(i: Tensor((), "float32"), s: Tensor((2, 3), "float32"), x: Tensor((2, 3), "float32")) -> Tensor(None, "float32", ndim = 2):
            # block 0
            cond: Tensor((), "bool") = R.call_packed("test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool")))
            if cond:
                # block 0
                another_cond: Tensor((), "bool") = R.call_packed("test.vm.less", i, relax.const(10), type_args=(Tensor(ndim=0, dtype="bool")))
                if another_cond:
                    # block 0
                    tensor_3: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
                    new_s1: Tensor((2, 3), "float32") = relax.add(s, tensor_3)
                    kill_tensor: Tuple() = relax.memory.kill_tensor(tensor_3)
                    new_s2 = relax.add(s, new_s1)
                else:
                    # block 0
                    tensor_4: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
                    tensor_5: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
                    new_s11: Tensor((2, 3), "float32") = relax.add(tensor_5, tensor_4)
                    kill_tensor1: Tuple() = relax.memory.kill_tensor(tensor_5)
                    kill_tensor2: Tuple() = relax.memory.kill_tensor(tensor_4)
                    new_s2 = new_s11
                output = new_s2
            else:
                # block 0
                tensor_1: Tensor((2, 3), "float32") = relax.builtin.alloc_tensor((2, 3), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
                new_s3: Tensor((2, 3), "float32") = relax.add(tensor_1, x)
                kill_tensor3: Tuple() = relax.memory.kill_tensor(tensor_1)
                output = new_s3
            return output
    # fmt: on

    before_mod = Before
    expected_mod = Expected
    _check(before_mod, expected_mod)


if __name__ == "__main__":
    pytest.main([__file__])
