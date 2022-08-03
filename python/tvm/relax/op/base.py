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
"""The base Relax operators."""
from typing import Union, List, Optional

import tvm
from tvm.runtime.object import Object

from . import _ffi_api
from ..expr import Expr, ShapeExpr, Tuple, Call
from ..ty import DynTensorType, TupleType
from ...ir import Array


def call_tir(
    func: Expr,
    args: Union[Tuple, List[Expr]],
    shape: Union[Tuple, ShapeExpr, List[int]],
    dtype: Union[str, List[str]],
    tir_vars: Optional[ShapeExpr] = None,
) -> Call:
    """
    Call a destination-passing-style function and return the output.

    Parameters
    ----------
    func : Expr
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Union[Tuple, List[Expr]]
        The input arguments.

    shape: Union[Tuple, ShapeExpr, List[int]]
        The output shape. Tuple(ShapeExpr) if multiple outputs, ShapeExpr if single output.

    dtype: Union[str, List[str]]
        The output dtype. List[str] if multiple outputs, str if single output.

    tir_vars : ShapeExpr, optional
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    if isinstance(shape, (list, tuple, Array)):
        shape = ShapeExpr(shape)

    if isinstance(args, (list, tuple)):
        args = Tuple(args)

    if isinstance(dtype, str):
        output_type = DynTensorType(len(shape), dtype)
    elif isinstance(dtype, (list, tuple)):
        if len(shape) != len(dtype):
            raise ValueError("The number of output_shape and output_dtype of call_tir mismatch")
        output_type = TupleType([DynTensorType(len(x), y) for x, y in zip(shape, dtype)])
    else:
        raise TypeError("Not supported dtype for call_tir: " + str(type(dtype)))

    return _ffi_api.call_tir(func, args, shape, output_type, tir_vars)


def make_closure(
    func: Expr,
    args: Union[Tuple, List[Expr]],
) -> Object:
    """
    Create a closure with free variables and return the closure.

    Parameters
    ----------
    func : Expr
        The closure, can be ExternFunc or PrimFunc.

    args : Union[Tuple, List[Expr]]
        The input arguments.


    Returns
    -------
    ret: Object
        The VMClosure.
    """

    if isinstance(args, (list, tuple)):
        args = Tuple(args)

    return _ffi_api.make_closure(func, args)


def invoke_closure(
    closure: Expr,
    args: Union[Tuple, List[Expr]],
) -> Object:
    """
    Invoke a closure.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Union[Tuple, List[Expr]]
        The input arguments.


    Returns
    -------
    ret: Object
        The result.
    """

    if isinstance(args, (list, tuple)):
        args = Tuple(args)

    return _ffi_api.invoke_closure(closure, args)


@tvm.register_func("relax.run.print")
def relax_print(val: tvm.runtime.container, format_str: str = "") -> None:
    """
    Prints the value that is passed to it, possibly with a format string.

    Parameters
    ----------
    val: tvm.Object
        A value in TVM

    format_str: str
        A Python-style format string for printing the value
    """

    def render(val: tvm.Object) -> str:
        if isinstance(val, tvm.runtime.ndarray.NDArray):
            return str(val)
        # no pretty-printer by default, so if we don't handle this,
        # then we can't look inside tuples
        if isinstance(val, tvm.runtime.container.ADT):
            # the fields of an ADT are not directly accessible in Python,
            # so I'm doing it this way
            fields = ", ".join([render(val[i]) for i in range(len(val))])
            # special case: tag = 0 is a tuple
            if val.tag == 0:
                return f"({fields})"
            return f"ADT(tag={val.tag}, fields=[{fields}])"
        return str(val)

    if format_str == "":
        print(render(val))
    else:
        print(format_str.format(render(val)))
