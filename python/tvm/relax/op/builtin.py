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
"""The builtin Relax operators."""

from . import _ffi_api
from ..expr import Expr, Call
from ...ir import make_node, Op


def shape_of(expr: Expr) -> Call:
    return _ffi_api.shape_of(expr)  # type: ignore # pylint: disable=no-member


def builtin_alloc_tensor(shape: Expr, dtype: str, runtime_device_index: int) -> Call:
    op = Op.get("relax.builtin.alloc_tensor")
    attrs = make_node(
        op.attrs_type_key,
        dtype=dtype,
        runtime_device_index=runtime_device_index,
    )
    return Call(op, (shape,), attrs=attrs)  # type: ignore # pylint: disable=no-member


def vm_builtin_alloc_storage(size: Expr) -> Call:
    return _ffi_api.vm.builtin.alloc_storage(size)  # type: ignore # pylint: disable=no-member


def vm_builtin_alloc_tensor(storage: Expr, shape: Expr) -> Call:
    return _ffi_api.vm.builtin.alloc_tensor(storage, shape)  # type: ignore # pylint: disable=no-member


def vm_builtin_store_shape(shape: Expr, heap: Expr) -> Call:
    return _ffi_api.vm.builtin.store_shape(shape, heap)  # type: ignore # pylint: disable=no-member


def vm_builtin_load_shape(heap: Expr) -> Call:
    return _ffi_api.vm.builtin.load_shape(heap)  # type: ignore # pylint: disable=no-member
