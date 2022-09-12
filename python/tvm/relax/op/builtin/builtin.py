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

from typing import List, Union
from tvm.ir.expr import PrimExpr
from . import _ffi_api
from ...expr import ShapeExpr, Call


# TODO(relax-team): add documents
def alloc_tensor(
    shape: Union[ShapeExpr, PrimExpr, List[PrimExpr]], dtype: str, runtime_device_index: int
) -> Call:
    if not isinstance(shape, ShapeExpr):
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ShapeExpr(shape)
    return _ffi_api.alloc_tensor(shape, dtype, runtime_device_index)
