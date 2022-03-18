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
# pylint: disable=invalid-name, unused-import
"""The type nodes of the Relax language."""
import tvm._ffi
from tvm.ir import Type, TensorType, TupleType, Span

from . import _ffi_api


@tvm._ffi.register_object("relax.ShapeType")
class ShapeType(Type):
    def __init__(self, span: Span = None):
        self.__init_handle_by_constructor__(_ffi_api.ShapeType, span)


@tvm._ffi.register_object("relax.DynTensorType")
class DynTensorType(Type):
    """A dynamic tensor type in Relax.

    This is the type assigned to tensors with a known dtype and unknown shape.

    Parameters
    ----------
    rank : Optional[int]
        The rank of the Tensor

    dtype : Optional[str]
        The content data type.
    """

    def __init__(self, rank=-1, dtype="float32", span: Span = None):
        self.__init_handle_by_constructor__(_ffi_api.DynTensorType, rank, dtype, span)


@tvm._ffi.register_object("relax.DimType")
class DimType(Type):
    """The type of indices/shape dimensions in Relax."""

    def __init__(self, span: Span = None):
        self.__init_handle_by_constructor__(_ffi_api.DimType, span)
