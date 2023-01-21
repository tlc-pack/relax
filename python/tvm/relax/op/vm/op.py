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

from typing import List, Tuple, Union

from tvm import DataType
from tvm.ir.expr import PrimExpr

from ...expr import Call, Expr, ExternFunc, ShapeExpr
from ...expr import Tuple as RxTuple
from ...utils import convert_to_expr
from . import _ffi_api


def alloc_storage(size: Expr, dtype: Union[DataType, str], runtime_device_index: int) -> Call:
    """Construct a Call to allocate storage with specific size, dtype, runtime_device_index.

    Parameters
    ----------
    size: Expr
        The size of the storage to be allocated.

    dtype: Union[DataType, str]
        The data type of the storage to be allocated.

    runtime_device_index : int
        The device index indicating on which device the storage is to be allocated at runtime.
        Index -1 is reserved for the host device.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated storage.
    """
    if isinstance(size, (tuple, list)):
        size = ShapeExpr(size)
    elif not isinstance(size, Expr):
        raise TypeError("size must be a tuple of PrimExpr or relax.Expr")
    return _ffi_api.alloc_storage(size, dtype, runtime_device_index)  # type: ignore


def alloc_tensor(
    storage: Expr,
    shape: Union[Tuple[PrimExpr], Expr],
    offset: int,
    dtype: Union[DataType, str],
) -> Call:
    """Construct a Call to allocate a tensor with specific shape, dtype, runtime_device_index.

    Parameters
    ----------
    storage: Expr
        The storage location to be allocated.

    shape: Union[Tuple[PrimExpr], Expr]
        The shape of the tensor to be allocated.

    offset: int
        The offset of the tensor to be allocated.

    dtype: Union[DataType, str]
        The datatype of the tensor to be allocated.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    elif not isinstance(shape, Expr):
        raise TypeError("storage must be a tuple of PrimExpr or relax.Expr")
    return _ffi_api.alloc_tensor(storage, shape, offset, dtype)  # type: ignore


def call_tir_dyn(
    func: Union[str, Expr],
    args: Union[Expr, List[Expr]],
) -> Call:
    """Construct a Call to call a tir function with dynamic shape.

    Parameters
    ----------
    func: Union[str, Expr],
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args: Union[Expr, List[Expr]]
        The arguments to the tir function.

    Returns
    -------
    result : Call
        A relax Call, which calls the tir function.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    args = convert_to_expr(args)

    return _ffi_api.call_tir_dyn(func, args)  # type: ignore
