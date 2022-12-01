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
from tvm import relax as rx, TVMError
from tvm.relax.ty import is_base_of, find_lca


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
    # check the subtype relation for DynTensorType
    # e.g., DynTensorType(ndim=3, dtype="float32") is a subtype of DynTensorType(ndim=-1, dtype="float32")
    # and DynTensorType(ndim=-1, "float32") is a subtype of DynTensorType(ndim=-1, dtype=None)
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
    # a type is subtype of itself
    assert is_base_of(t2, t2)
    assert is_base_of(t3, t3)
    assert is_base_of(t2, t3) == False
    assert is_base_of(t3, t2) == False

    # check the subtype relation for ShapeType
    t5 = rx.ShapeType()
    t6 = rx.ShapeType()
    assert is_base_of(t5, t5)
    assert is_base_of(t5, t6)
    assert is_base_of(t5, t0) == False

    # check the subtype relation for TupleType by checking if each field
    # of the base TupleType is subtype of the field of the derived TupleType
    # e.g., TupleType([DynTensorType(ndim=3, dtype="float32"), ShapeType()])
    # is a subtype of TupleType([DynTensorType(ndim=-1, dtype="float32"), ShapeType()])
    t7 = rx.TupleType([t0, t1, t5])
    t8 = rx.TupleType([t1, t1, t5])
    t9 = rx.TupleType([t1, t3, t5])
    t10 = rx.TupleType([t5, t3, t1])
    t11 = rx.TupleType([t1, t3])
    assert is_base_of(t7, t8)
    assert is_base_of(t7, t9)
    assert is_base_of(t8, t9)
    assert is_base_of(t8, t8)
    assert is_base_of(t9, t7) == False
    assert is_base_of(t7, t10) == False
    assert is_base_of(t11, t7) == False
    assert is_base_of(t7, t11) == False

    # check the subtype relation for FunctionType by checking the subtype relations of arg_types and ret_type
    # e.g., FuncType([DynTensorType(ndim=3, dtype="float32")], DynTensorType(ndim=2, dtype="float32"))
    # is a subtype of FuncType([DynTensorType(ndim=-1, dtype=None)], DynTensorType(ndim=-1, dtype="float32"))
    t12 = rx.FuncType([t7], t0)
    t13 = rx.FuncType([t7], t1)
    t14 = rx.FuncType([t8], t0)
    t15 = rx.FuncType([t8], t1)
    t16 = rx.FuncType([t7, t0], t1)
    t17 = rx.FuncType([t7, t4], t1)
    assert is_base_of(t12, t13)
    assert is_base_of(t12, t14)
    assert is_base_of(t12, t15)
    assert is_base_of(t12, t12)
    assert is_base_of(t13, t14) == False
    assert is_base_of(t13, t15)
    assert is_base_of(t14, t15)
    assert is_base_of(t16, t17)
    assert is_base_of(t16, t16)
    assert is_base_of(t12, t16) == False
    assert is_base_of(t13, t16) == False

    # check the subtype relation for ObjectType
    # ObjectType is the base type of every type in Relax
    t18 = rx.ObjectType()
    assert is_base_of(t18, t0)
    assert is_base_of(t18, t5)
    assert is_base_of(t18, t7)
    assert is_base_of(t18, t12)
    assert is_base_of(t18, t18)
    assert is_base_of(t0, t18) == False
    assert is_base_of(t5, t18) == False
    assert is_base_of(t7, t18) == False
    assert is_base_of(t12, t18) == False

    # more complicated cases
    # TupleType with all possible types as fields
    t19 = rx.TupleType([t7, t0, t5, t12, t18])
    t20 = rx.TupleType([t8, t1, t5, t15, t18])
    t21 = rx.TupleType([t18, t18, t18, t18, t18])
    assert is_base_of(t19, t20)
    assert is_base_of(t21, t19)
    assert is_base_of(t21, t20)
    assert is_base_of(t20, t19) == False
    assert is_base_of(t18, t20)
    # FuncType with all possible types as arg_types and ret_type
    t22 = rx.FuncType([t7, t0, t5, t12, t18], t0)
    t23 = rx.FuncType([t8, t1, t5, t15, t18], t1)
    t24 = rx.FuncType([t7], t0)
    t25 = rx.FuncType([t18, t18, t18, t18, t18], t18)
    t26 = rx.FuncType([t18], t18)
    t27 = rx.FuncType([t7, t0, t5, t12, t18], t19)
    t28 = rx.FuncType([t7, t0, t5, t12, t18], t20)
    assert is_base_of(t22, t23)
    assert is_base_of(t25, t23)
    assert is_base_of(t18, t23)
    assert is_base_of(t18, t22)
    assert is_base_of(t27, t28)
    assert is_base_of(t24, t22) == False
    assert is_base_of(t24, t23) == False
    assert is_base_of(t26, t23) == False
    assert is_base_of(t28, t27) == False

    # check the subtype relation for PackedFuncType
    t29 = rx.PackedFuncType()
    t30 = rx.PackedFuncType()
    assert is_base_of(t29, t30)
    assert is_base_of(t30, t29)
    assert is_base_of(t18, t29)
    assert is_base_of(t1, t29) == False


def test_lca():
    t0 = rx.DynTensorType(-1, None)
    t1 = rx.DynTensorType(3, None)
    t2 = rx.DynTensorType(3, "int32")
    t3 = rx.DynTensorType(3, "float32")
    t4 = rx.DynTensorType(3, "float32")
    assert find_lca(t0, t1) == t0
    assert find_lca(t0, t2) == t0
    assert find_lca(t0, t3) == t0
    assert find_lca(t0, t4) == t0
    assert find_lca(t1, t2) == t1
    assert find_lca(t1, t3) == t1
    assert find_lca(t4, t3) == t3

    t5 = rx.ShapeType()
    t6 = rx.ShapeType()
    assert find_lca(t5, t5) == t5
    assert find_lca(t5, t6) == t5

    t7 = rx.TupleType([t0, t1, t5])
    t8 = rx.TupleType([t1, t1, t5])
    t9 = rx.TupleType([t1, t3, t5])
    t10 = rx.TupleType([t5, t3, t1])
    t11 = rx.TupleType([t1, t3])
    assert find_lca(t7, t8) == t7
    assert find_lca(t7, t9) == t7
    assert find_lca(t8, t9) == t8
    assert find_lca(t8, t8) == t8
    assert find_lca(t7, t10) == rx.TupleType([rx.ObjectType(), t1, rx.ObjectType()])
    assert find_lca(t7, t11) == rx.ObjectType()

    t12 = rx.FuncType([t7], t0)
    t13 = rx.FuncType([t7], t1)
    t14 = rx.FuncType([t8], t0)
    t15 = rx.FuncType([t8], t1)
    t16 = rx.FuncType([t7, t0], t1)
    t17 = rx.FuncType([t7, t4], t1)
    assert find_lca(t12, t13) == t12
    with pytest.raises(TVMError):
        find_lca(t12, t14)
        find_lca(t16, t17)
    assert find_lca(t14, t15) == t14

    t18 = rx.ObjectType()
    assert find_lca(t18, t0) == t18
    assert find_lca(t18, t5) == t18
    assert find_lca(t18, t7) == t18
    assert find_lca(t18, t12) == t18
    assert find_lca(t18, t18) == t18

    t19 = rx.PackedFuncType()
    assert find_lca(t19, t0) == t18
    assert find_lca(t19, t12) == t18


if __name__ == "__main__":
    pytest.main([__file__])
