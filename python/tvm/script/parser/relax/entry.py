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
# pylint: disable=missing-docstring, invalid-name
import inspect
from typing import Callable as _Callable
from typing import List, Optional, Tuple
from typing import TypeVar as _TypeVar
from typing import Union

from tvm import relax
from tvm.relax import DynTensorType, Expr, Function, StructInfo
from tvm.relax import Tuple as RxTuple
from tvm.relax import Type, Var
from tvm.runtime import ObjectGeneric
from tvm.tir import PrimExpr

from ...ir_builder.relax import tensor
from .._core import parse, utils

FType = _TypeVar("FType", bound=_Callable)


def function(f: FType) -> Union[Function, FType]:
    if not inspect.isfunction(f):
        raise TypeError(f"Expect a function, but got: {f}")
    if utils.is_defined_in_class(inspect.stack(), f):
        return f
    return parse(f, utils.inspect_function_capture(f))


setattr(function, "dispatch_token", "relax")


############################### R.Tensor ###############################


class TensorProxy(ObjectGeneric):
    def __call__(
        self,
        shape: Optional[List[Union[PrimExpr, str]]] = None,
        dtype: str = None,
        ndim: int = -1,
    ) -> relax.TensorStructInfo:
        # scalar tensor case
        if shape is not None and len(shape) == 0:
            shape = []
        if isinstance(shape, str) and dtype is None:
            dtype = shape
            shape = None
        return tensor(shape, dtype, ndim)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore

    def asobject(self):
        """Convert to object when direct call `R.Tensor`
        e.g. `x = R.invoke_closure(clo, (y,), type_args=R.Tensor)`
        """
        return DynTensorType()


Tensor = TensorProxy()  # pylint: disable=invalid-name

############################## R.Callable ##############################


class CallableProxy:
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    Parameters
    ----------
    params : List[StructInfo]
        The argument StructInfo

    ret : StructInfo
        The return StructInfo.

    """

    def __call__(
        self,
        params: Union[StructInfo, List[StructInfo], Tuple[StructInfo]],
        ret: StructInfo,
    ) -> relax.FuncStructInfo:
        if not isinstance(params, (list, tuple)):
            params = [params]
        return relax.FuncStructInfo(params, ret)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Callable = CallableProxy()

############################### R.Tuple ################################


class TupleProxy:
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Union[Expr, Type, StructInfo]]
        The fields in the tuple
    """

    def __call__(
        self,
        *fields: List[Union[Expr, Type, StructInfo]],
    ) -> Union[Expr, StructInfo]:
        if len(fields) == 1 and isinstance(fields[0], (tuple, list)):
            fields = fields[0]

        # TODO(siyuan): Revisit this part
        if all([isinstance(f, Expr) for f in fields]):
            return RxTuple(fields)
        else:
            fields = list(fields)
            for i, x in enumerate(fields):
                if callable(x):
                    fields[i] = x()
            if all([isinstance(f, StructInfo) for f in fields]):
                return relax.TupleStructInfo(fields)
            else:
                raise TypeError(f"Invalid tuple type: {fields}")

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Tuple = TupleProxy()

############################### R.Shape ################################


class ShapeProxy:
    """The type of shape values.

    Parameters
    ----------
    values : Optional[List[PrimExpr]]
       The symbolic shape values if known.

    ndim : Optional[int]
       The size of the shape.
    """

    def __call__(
        self,
        values: Optional[List[PrimExpr]] = None,
        ndim: int = -1,
    ) -> StructInfo:
        return relax.ShapeStructInfo(values, ndim)

    def __getitem__(self, keys) -> Var:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Shape = ShapeProxy()

############################ R.match_shape #############################
class MatchShapePair:
    value: Expr
    pattern: List[PrimExpr]

    def __init__(self, value: Expr, pattern: List[PrimExpr]) -> None:
        self.value = value
        self.pattern = pattern


def match_shape(value: Expr, pattern: List[PrimExpr]):
    if value is None:
        raise ValueError("value of match_shape cannot be None")
    if pattern is None:
        raise ValueError("pattern of match_shape cannot be None")
    return MatchShapePair(value, pattern)


############################ R.match_cast #############################
class MatchCastPair:
    value: Expr
    struct_info: StructInfo

    def __init__(self, value: Expr, struct_info: StructInfo) -> None:
        self.value = value
        self.struct_info = struct_info


def match_cast(value: Expr, struct_info: StructInfo):
    if value is None:
        raise ValueError("value of match_cast cannot be None")
    if struct_info is None:
        raise ValueError("struct_info of match_cast cannot be None")
    return MatchCastPair(value, struct_info)
