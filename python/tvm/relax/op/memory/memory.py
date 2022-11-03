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
"""Relax memory primitives."""

from . import _ffi_api
from ...expr import Expr, Call


def alloc_storage(size: Expr, virtual_device_index: int, storage_scope: str, dtype: str) -> Call:
    """Construct a Call to allocate a storage with specific size, virtual_device_index,
    storage_scope and dtype.

    Parameters
    ----------
    size : Expr
        The size of the storage to be allocated.

    virtual_device_index : int
        The virtual device index indicating on which device the storage is to be allocated.
        Index -1 is reserved for the host device.

    storage_scope : str
        The storage scope to allocate the storage to.

    dtype : str
        The datatype of the storage to be allocated.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated storage.
    """
    return _ffi_api.alloc_storage(size, virtual_device_index, storage_scope, dtype)


def alloc_tensor(storage: Expr, shape: Expr, offset: int, dtype: str) -> Call:
    """Construct a Call to allocate a tensor on a certain storage starting from the given offset.

    Parameters
    ----------
    storage : Expr
        The storage to allocate the tensor to.

    shape : Expr
        The shape of the tensor to be allocated.

    offset : int
        The storage offset to allocate the tensor.

    dtype : str
        The datatype of the tensor to be allocated.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated tensor.
    """
    return _ffi_api.alloc_tensor(storage, shape, offset, dtype)


def kill_storage(storage: Expr) -> None:
    """Construct a Call to kill a storage.

    Parameters
    ----------
    storage : Expr
        The storage to be killed.
    """
    return _ffi_api.kill_storage(storage)


def kill_tensor(tensor: Expr) -> None:
    """Construct a Call to kill a tensor.

    Parameters
    ----------
    tensor : Expr
        The tensor to be killed.
    """
    return _ffi_api.kill_tensor(tensor)
