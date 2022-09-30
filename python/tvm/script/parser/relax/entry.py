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
from typing import Callable as _Callable
from typing import List, Optional
from typing import TypeVar as _TypeVar
from typing import Union

from tvm.ir import FuncType, TypeConstraint, TypeVar
from tvm.relax import Expr, Function, Type, Var
from tvm.tir import PrimExpr

from ...ir_builder.relax import TensorType, tensor
from .._core import parse, utils
from ..ir import is_defined_in_class

FType = _TypeVar("FType", bound=_Callable)


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
        return tensor(shape, dtype, ndim)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Tensor = TensorProxy()  # pylint: disable=invalid-name


class CallableProxy:
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    We can informally write them as:
    `forall (type_params), (arg_types) -> ret_type where type_constraints`

    Parameters
    ----------
    arg_types : List[Type]
        The argument types

    ret_type : Type
        The return type.

    type_params : Optional[List[TypeVar]]
        The type parameters

    type_constraints : Optional[List[TypeConstraint]]
        The type constraints.
    """

    def __call__(
        self,
        arg_types: List[Type],
        ret_type: Type,
        type_params: Optional[List[TypeVar]] = None,
        type_constraints: Optional[List[TypeConstraint]] = None,
    ) -> FuncType:
        def _convert_type(ty: Union[Type, TensorType]) -> Type:
            if isinstance(ty, TensorType):
                return ty.type
            elif isinstance(ty, Type):
                return ty
            else:
                raise TypeError(f"Expect a Type or TensorType, but got: {ty}")

        arg_types = [_convert_type(ty) for ty in arg_types]
        ret_type = _convert_type(ret_type)
        return FuncType(arg_types, ret_type, type_params, type_constraints)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Callable = CallableProxy()


class MatchShapePair:
    value: Expr
    pattern: List[PrimExpr]

    def __init__(self, value: Expr, pattern: List[PrimExpr]) -> None:
        self.value = value
        self.pattern = pattern


def match_shape(value: Expr, pattern: List[PrimExpr]):
    return MatchShapePair(value, pattern)
