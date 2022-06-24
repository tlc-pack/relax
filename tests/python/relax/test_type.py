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
from tvm.relax.ty import is_base_of


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


def test_subtype():
    # DynTensorType
    t0 = rx.DynTensorType(-1, None)
    t1 = rx.DynTensorType(3, None)
    t2 = rx.DynTensorType(3, "int32")
    t3 = rx.DynTensorType(3, "float32")
    t4 = rx.DynTensorType(3, "float32")
    assert is_base_of(t0, t1)
    assert is_base_of(t0, t2)
    assert is_base_of(t0, t3)
    assert is_base_of(t0, t4)
    assert is_base_of(t1, t2)
    assert is_base_of(t1, t3)
    assert is_base_of(t4, t3)
    assert is_base_of(t2, t3) == False
    assert is_base_of(t3, t2) == False

    # ShapeType
    t5 = rx.ShapeType()
    t6 = rx.ShapeType()
    assert is_base_of(t5, t6)
    assert is_base_of(t5, t0) == False

    # TupleType
    t7 = rx.TupleType([t0, t1, t5])
    t8 = rx.TupleType([t1, t1, t5])
    t9 = rx.TupleType([t1, t3, t5])
    t10 = rx.TupleType([t5, t3, t1])
    t11 = rx.TupleType([t1, t3])
    assert is_base_of(t7, t8)
    assert is_base_of(t7, t9)
    assert is_base_of(t8, t9)
    assert is_base_of(t9, t7) == False
    assert is_base_of(t7, t10) == False
    assert is_base_of(t11, t7) == False
    assert is_base_of(t7, t11) == False

    # FunctionType
    t12 = rx.FuncType([t7], t0)
    t13 = rx.FuncType([t7], t1)
    t14 = rx.FuncType([t8], t0)
    t15 = rx.FuncType([t8], t1)
    t16 = rx.FuncType([t7, t0], t1)
    t17 = rx.FuncType([t7, t4], t1)
    assert is_base_of(t12, t13)
    assert is_base_of(t12, t14)
    assert is_base_of(t12, t15)
    assert is_base_of(t13, t14) == False
    assert is_base_of(t13, t15)
    assert is_base_of(t14, t15)
    assert is_base_of(t16, t17)
    assert is_base_of(t12, t16) == False
    assert is_base_of(t13, t16) == False

    # ObjectType
    t18 = rx.ObjectType()
    assert is_base_of(t18, t0)
    assert is_base_of(t18, t5)
    assert is_base_of(t18, t7)
    assert is_base_of(t18, t12)
    assert is_base_of(t0, t18) == False
    assert is_base_of(t5, t18) == False
    assert is_base_of(t7, t18) == False
    assert is_base_of(t12, t18) == False


if __name__ == "__main__":
    pytest.main([__file__])
