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
# pylint: disable=redefined-builtin
"""The base Relax operators."""
from typing import Union, List, Optional


import tvm
from tvm.runtime.object import Object

from . import _ffi_api
from ..expr import Expr, ShapeExpr, Call, ExternFunc
from ..expr import Tuple as RxTuple
from ..ty import DynTensorType, TupleType
from ...ir import Array, Type, PrimExpr


py_print = print  # pylint: disable=invalid-name


def null_value() -> Call:
    """Create a call node that represents a null value object.

    Returns
    -------
    ret: Call
        The created call node.
    """
    return _ffi_api.null_value()  # type: ignore


def call_tir(
    func: Union[str, Expr],
    args: Union[Expr, List[Expr]],
    shape: Union[RxTuple, ShapeExpr, List[int]],
    dtype: Union[str, List[str]],
    tir_vars: Optional[ShapeExpr] = None,
) -> Call:
    """
    Call a destination-passing-style function and return the output.

    Parameters
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Union[Expr, List[Expr]]
        The input arguments.

    shape: Union[RxTuple, ShapeExpr, List[int]]
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

    def _create_shape(shape: List[Union[int, PrimExpr]]) -> ShapeExpr:
        shape_array = []
        for x in shape:
            if isinstance(x, int):
                shape_array.append(tvm.tir.IntImm("int64", x))
            elif isinstance(x, tvm.tir.IntImm):
                shape_array.append(x if x.dtype == "int64" else tvm.tir.IntImm("int64", x.value))
            elif isinstance(x, PrimExpr):
                if x.dtype != "int64":
                    raise TypeError("Expect int64 dtype for shape")
                shape_array.append(x)
            else:
                raise TypeError("Expect int or PrimExpr for shape")
        return ShapeExpr(shape_array)

    if isinstance(shape, (list, tuple, Array)):
        if all([not isinstance(x, (list, tuple, Array, ShapeExpr)) for x in shape]):
            shape = _create_shape(shape)  # type: ignore
        elif all([isinstance(x, (list, tuple, Array, ShapeExpr)) for x in shape]):
            shape = RxTuple(
                [
                    _create_shape(x) if not isinstance(x, ShapeExpr) else x  # type: ignore
                    for x in shape
                ]
            )
        else:
            raise TypeError(
                f"The shape is expected to be ShapeExpr or Tuple[ShapeExpr], bot got: f{shape}"
            )

    if isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if isinstance(args, (list, tuple)):
        args = RxTuple(args)

    if isinstance(dtype, str):
        output_type = DynTensorType(len(shape), dtype)
    elif isinstance(dtype, (list, tuple)):
        if len(shape) != len(dtype):
            raise ValueError("The number of output_shape and output_dtype of call_tir mismatch")
        output_type = TupleType([DynTensorType(len(x), y) for x, y in zip(shape, dtype)])
    else:
        raise TypeError("Not supported dtype for call_tir: " + str(type(dtype)))

    return _ffi_api.call_tir(func, args, shape, output_type, tir_vars)  # type: ignore


def call_builtin(
    func: Union[str, Expr],
    args: Union[RxTuple, List[Expr]],
    *,
    type_args: Optional[Union[Type, List[Type]]] = None,
    int_args: Optional[List[int]] = None,
    dtype_arg: Optional[str] = None,
    str_args: Optional[List[str]] = None,
    require_ctx: bool = False,
) -> Call:
    """Call a builtin function func.

    Parameters
    ----------
    func : Expr
        The builtin function to be called.

    args : Union[RxTuple, List[Expr]]
        The input arguments.

    type_args: Optional[Union[Type, List[Type]]]
        The type arguments to the call node.

    int_args: Optional[List[int]]
        List of additional int arguments passed to the builtin.

    dtype_arg: str
        Additional dtype argument passed to the builtin

    str_args:  Optional[List[str]]
        List of additional int arguments passed to the builtin.

    require_ctx: bool
        Whether we need to pass context as first argument.

    Returns
    -------
    ret: Call
        The created call node.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    if isinstance(args, (list, tuple)):
        args = RxTuple(args)

    if type_args is not None and not isinstance(type_args, (list, tuple)):
        type_args = [type_args]

    return _ffi_api.call_builtin(  # type: ignore
        func, args, type_args, int_args, dtype_arg, str_args, require_ctx  # type: ignore
    )


def make_closure(
    func: Expr,
    args: Union[RxTuple, List[Expr]],
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
        args = RxTuple(args)

    return _ffi_api.make_closure(func, args)  # type: ignore


def invoke_closure(
    closure: Expr,
    args: Union[RxTuple, List[Expr]],
    type_args: Union[List[Type], Type],
) -> Object:
    """
    Invoke a closure.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Union[Tuple, List[Expr]]
        The input arguments.

    type_args: Union[Tuple[Type], Type]
        The type_args of the CallNode

    Returns
    -------
    ret: Object
        The result.
    """

    if isinstance(args, (list, tuple)):
        args = RxTuple(args)
    if not isinstance(type_args, (list, tuple)):
        type_args = (type_args,)

    return _ffi_api.invoke_closure(closure, args, type_args)  # type: ignore


def render_object(val: tvm.Object) -> str:
    """
    Given a TVM Object, renders it in string form. Used for Relax printing and assertions.

    Parameters
    ----------
    val: tvm.Object
        An object to render

    Returns
    -------
    ret: str
        A string representing the value, ideally human-readable
    """
    if isinstance(val, tvm.runtime.ndarray.NDArray):
        return str(val)
    # no pretty-printer by default, so if we don't handle this,
    # then we can't look inside tuples
    if isinstance(val, tvm.runtime.container.ADT):
        # the fields array of an ADT cannot be directly accessed in Python
        # so we have to get the length and index into the fields separately
        fields = ", ".join([render_object(val[i]) for i in range(len(val))])
        # special case: tag = 0 is a tuple
        if val.tag == 0:
            return f"({fields})"
        return f"ADT(tag={val.tag}, fields=[{fields}])"
    return str(val)


@tvm.register_func("relax.run.print")
def relax_print(format_str: str, *format_args: tvm.Object) -> None:
    """
    Takes a list of values to print, formats with the given format string.
    If the format string is empty, simply prints.

    Call from TVM script like this:
    `relax.print(value1, value2, ..., valueN, format=format_str)`
    or
    `relax.print(value1, value2, ..., valueN) # format_str defaults to ""`

    Parameters
    ----------
    format_str: str
        The last argument is a Python-style format string for printing the value

    format_args: List[Object]
        The values to print.
    """
    val_strs = map(render_object, format_args)
    if format_str == "":
        py_print(*val_strs)
    else:
        py_print(format_str.format(*val_strs))


def print(*values: List[Expr], format: str = "") -> Expr:
    """Print op to print the values

    Parameters
    ----------
    values : List[Expr]
        The values to print.

    format_str: str
        The format string.

    Returns
    -------
    result : Expr
        A relax Call, which will print the value during runtime.
    """
    return _ffi_api.print(values, format)  # type: ignore # pylint: disable=no-member


@tvm.register_func("relax.run.assert_op")
def relax_assert_op(condition: tvm.Object, format_str: str, *format_args: tvm.Object) -> None:
    """
    A variadic function. The first value serves as the assertion condition:
    If the condition is true, then the operator does nothing.
    If the condition is false, then the operator raises an assertion error.

    Arguments after the first value serve as format arguments for the error message;
    the last argument must be a format string for the error message (empty by default).
    If the format string is the empty string, then the error message will simply include
    a comma-separated list of the format arguments.
    The condition argument is not included in the format string.

    Parameters
    ----------
    condition: tvm.Object
        The assertion condition. Must be a boolean scalar.

    format_str: str
        The last argument is a Python-style format string for printing the value

    format_args: List[tvm.Object]
        Values used for formatting the string.
    """
    if not isinstance(format_str, str):
        raise ValueError(
            f"The format string argument to assert must be a string, given {type(format_str)})"
        )

    # should be guaranteed by the type system
    if not isinstance(condition, tvm.runtime.ndarray.NDArray):
        raise ValueError(f"The condition must be an NDArray, but given a {type(condition)}.")

    # may happen if the original program had unknown shape or dtype for the tensor's type
    dtype = condition.dtype
    if dtype != "bool":
        raise ValueError(f"The condition must be a bool scalar, but given a {dtype} tensor")
    shape = condition.shape
    if len(shape) != 0:
        raise ValueError(f"The condition must be a scalar, but it has a shape of {shape}")

    val = condition.numpy()
    if not val:
        error_message = "Assertion Failed"
        if format_args or format_str != "":
            rendered = map(render_object, format_args)
            if format_str != "":
                error_message = format_str.format(*rendered)
            else:
                error_message = ", ".join(rendered)
        raise AssertionError(error_message)


def assert_op(
    condition: Expr, format_args: Optional[Union[Expr, List[Expr]]] = None, format: str = ""
) -> Expr:
    """
    Create a call to Relax's assert_op operation (`assert` is reserved in Python,
    so the name must be distinct).

    Parameters
    ----------
    condition: Expr
        The assertion condition.

    format_args: Optional[Union[Expr, List[Expr]]]
        Format arguments for the error message if the condition fails.

    format_str: str
        The format string for the error message.

    Returns
    -------
    result : Expr
        A Call to the Relax assert operation.
    """
    if format_args is None:
        format_args = []
    if isinstance(format_args, Expr):  # type: ignore
        format_args = [format_args]
    return _ffi_api.assert_op(condition, format_args, format)  # type: ignore


def shape_of(expr: Expr) -> Expr:
    """Get shape of a tensor.

    Parameters
    ----------
    expr : Expr
        The input Expr.

    Returns
    -------
    result : Expr
        A relax Call, which gets the shape of the input
    """
    return _ffi_api.shape_of(expr)  # type: ignore # pylint: disable=no-member
