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
from ..expr import Expr, ShapeExpr, Tuple, Call, ExternFunc
from ..ty import DynTensorType, TupleType
from ...ir import Array


def call_tir(
    func: Union[str, Expr],
    args: Union[Expr, Tuple, List[Expr]],
    shape: Union[Tuple, ShapeExpr, List[int]],
    dtype: Union[str, List[str]],
    tir_vars: Optional[ShapeExpr] = None,
) -> Call:
    """
    Call a destination-passing-style function and return the output.

    Parameters
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Union[Expr, Tuple, List[Expr]]
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
    if isinstance(func, str):
        func = ExternFunc(func)

    if isinstance(shape, (list, tuple, Array)):
        shape = ShapeExpr(shape)

    if isinstance(args, Expr):
        args = Tuple((args,))

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
def relax_print(*args: List[any]) -> None:
    """
    Takes a list of values to print, formats with the given format string.
    If the format string is empty, simply prints.

    Since this function is called as a PackedFunc from the generated code,
    we cannot have it be variadic _and_ have an optional format string attribute
    except by taking in all the arguments as a single list. The last argument
    should be a format string.

    Call from TVM script like this:
    `relax.print(value1, value2, ..., valueN, format=format_str)`
    or
    `relax.print(value1, value2, ..., valueN) # format_str defaults to ""`

    Parameters
    ----------
    vals: List[Object]
        The values to print.

    format_str: str
        The last argument is a Python-style format string for printing the value
    """

    # there is no way to have a keyword arg to a packed function,
    # so the format string is always the last argument
    format_str = args[-1]
    if not isinstance(format_str, str):
        raise ValueError("No valid format string given.")

    def render(val: tvm.Object) -> str:
        if isinstance(val, tvm.runtime.ndarray.NDArray):
            return str(val)
        # no pretty-printer by default, so if we don't handle this,
        # then we can't look inside tuples
        if isinstance(val, tvm.runtime.container.ADT):
            # the fields array of an ADT cannot be directly accessed in Python
            # so we have to get the length and index into the fields separately
            fields = ", ".join([render(val[i]) for i in range(len(val))])
            # special case: tag = 0 is a tuple
            if val.tag == 0:
                return f"({fields})"
            return f"ADT(tag={val.tag}, fields=[{fields}])"
        return str(val)

    val_strs = map(render, args[:-1])
    if format_str == "":
        print(*val_strs)
    else:
        print(format_str.format(*val_strs))
