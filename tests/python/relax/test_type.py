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
import numpy as np
import tvm
from tvm import relax as rx


def test_shape_type():
    t0 = rx.ShapeType()
    t1 = rx.ShapeType()
    assert t0 == t1


def test_dyn_tensor_type():
    t0 = rx.DynTensorType()
    assert t0.ndim == -1
    t1 = rx.DynTensorType(3, "int32")
    assert t1.ndim == 3
    assert t1.dtype == "int32"


if __name__ == "__main__":
    pytest.main([__file__])
