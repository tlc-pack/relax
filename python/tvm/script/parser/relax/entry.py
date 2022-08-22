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
# pylint: disable=missing-docstring
import inspect
from typing import Callable, List, Optional, Union, TypeVar

from tvm.relax import Function, Var
from tvm.tir import PrimExpr

from ...ir_builder.relax import tensor_decl, TensorType
from .._core import parse, utils
from ..ir import is_defined_in_class


FType = TypeVar("FType", bound=Callable)


def function(f: FType) -> Union[Function, FType]:
    if not inspect.isfunction(f):
        raise TypeError(f"Expect a function, but got: {f}")
    if is_defined_in_class(inspect.stack()):
        return f
    return parse(f, utils.inspect_function_capture(f))


setattr(function, "dispatch_token", "relax")


class TensorProxy:
    def __call__(
        self,
        shape: Optional[List[Union[PrimExpr, str]]] = None,
        dtype: str = None,
        ndim: int = -1,
    ) -> TensorType:
        return tensor_decl(shape, dtype, ndim)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Tensor = TensorProxy()  # pylint: disable=invalid-name
