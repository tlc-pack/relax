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
"""The struct info nodes of the Relax language."""
from typing import List, Optional, Tuple, Union

import tvm._ffi
import tvm

from tvm.ir import Span, Node, EnvFunc, Array, Type
from tvm.tir import PrimExpr
from .expr import Var, Expr, ShapeExpr

from . import _ffi_api, ty, expr


class StructInfo(Node):
    """The base class of all StructInfo.

    StructInfo contains both the static type
    and runtime structural information.
    """

    def __eq__(self, other):
        """Compare two struct info for structural equivalence."""
        return bool(tvm.ir.structural_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Overload with structural equality."""
        return super().__eq__(other)

    def is_base_of(self, derived: "StructInfo") -> bool:
        """Check if self is base of another derived struct info.

        Parameters
        ----------
        derived : StructInfo
            The derived struct info to be checked.

        Returns
        -------
        result : bool
            The check result.
        """
        return _ffi_api.StructInfoIsBaseOf(self, derived)  # type: ignore


@tvm._ffi.register_object("relax.ObjectStructInfo")
class ObjectStructInfo(StructInfo):
    """StructInfo of an Object."""

    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ObjectStructInfo, span)  # type: ignore


@tvm._ffi.register_object("relax.PrimStructInfo")
class PrimStructInfo(StructInfo):
    """StructInfo of a primtive POD value.

    Parameters
    ----------
    dtype : str
       The data type of the prim value.
    """

    def __init__(self, dtype: str, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.PrimStructInfo, dtype, span)  # type: ignore


@tvm._ffi.register_object("relax.ShapeStructInfo")
class ShapeStructInfo(StructInfo):
    """StructInfo of a shape value.

    Parameters
    ----------
    values : Optional[List[PrimExpr]]
       The symbolic shape values if known.

    ndim : Optional[int]
       The size of the shape.
    """

    def __init__(
        self, values: Optional[List[PrimExpr]] = None, ndim: int = -1, span: Span = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ShapeStructInfo, values, ndim, span  # type: ignore
        )


@tvm._ffi.register_object("relax.TensorStructInfo")
class TensorStructInfo(StructInfo):
    """StructInfo of a Tensor value.

    Parameters
    ----------
    shape : Optional[Expr]
       The shape expression.

    dtype : Optional[str]
        The content data type.

    ndim : Optional[int]
       The number of dimensions of the tensor.
    """

    shape: Optional[Expr]
    dtype: tvm.DataType
    ndim: int
    span: Span

    def __init__(
        self,
        shape: Union[Optional[Expr], List[PrimExpr]] = None,
        dtype: str = "float32",
        ndim: int = -1,
        span: Span = None,
    ) -> None:
        if isinstance(shape, (list, tuple, Array)):
            shape = ShapeExpr(shape)

        self.__init_handle_by_constructor__(
            _ffi_api.TensorStructInfo, shape, dtype, ndim, span  # type: ignore
        )


@tvm._ffi.register_object("relax.TupleStructInfo")
class TupleStructInfo(StructInfo):
    """StructInfo of a Tuple value.

    Parameters
    ----------
    fields: List[StructInfo]
        The struct info of the fields.
    """

    def __init__(self, fields: List[StructInfo], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TupleStructInfo, fields, span)  # type: ignore


@tvm._ffi.register_object("relax.FuncStructInfo")
class FuncStructInfo(StructInfo):
    """StructInfo of a function value.

    Parameters
    ----------
    params: List[StructInfo]
        The struct info of the fields.

    ret: StructInfo
        The struct info of return valeu
    """

    def __init__(self, params: List[StructInfo], ret: StructInfo, span: Span = None) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.FuncStructInfo, params, ret, span  # type: ignore
        )

    @staticmethod
    def opaque_func(
        *,
        ret: Optional[StructInfo] = None,
        derive_func: Optional[EnvFunc] = None,
        span: Span = None,
    ) -> "FuncStructInfo":
        """
        Create an opaque FuncStructInfo

        Parameters
        ----------
        ret: Optional[StructInfo]
           The return value of the function.

        derive_func: Optional[EnvFunc]
           The environment function used for derivation

        span: Optional[Span]
           Optional span information of the ast.

        Returns
        -------
        info: FuncStructInfo
        """
        return _ffi_api.FuncStructInfoOpaqueFunc(ret, derive_func, span)  # type: ignore
