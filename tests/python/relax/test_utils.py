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
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import relax as R


def test_copy_with_new_params():
    @R.function
    def before(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
        gv = R.add(x, y)
        return gv

    after = relax.utils.copy_with_new_params(before)
    assert_structural_equal(after, before)

    assert len(after.params) == len(before.params)
    for before_var, after_var in zip(before.params, after.params):
        assert before_var != after_var


if __name__ == "__main__":
    pytest.main([__file__])
