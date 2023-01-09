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
"""Relax Neural Network (NN) operators"""
from typing import Optional, Tuple, Union

from tvm import DataType
from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ...expr import Expr

PrimExprLike = Union[int, PrimExpr]


def conv2d(
    data: Expr,
    weight: Expr,
    strides: Union[PrimExprLike, Tuple[PrimExprLike]] = (1, 1),
    padding: Union[PrimExprLike, Tuple[PrimExprLike]] = (0, 0),
    dilation: Union[PrimExprLike, Tuple[PrimExprLike]] = (1, 1),
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: Optional[str] = None,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_h, kernel_w)`,
    where `kernel_h` and `kernel_w` is the lengths of the `H` and `W` kernel dimensions,
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, y, x] = \sum_{dy, dx, k}
           \mbox{data}[b, k, \mbox{strides}[0] * y  + dy, \mbox{strides}[1] * x + dx] *
           \mbox{weight}[c, k, dy, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCHW` for data and `OIHW` for weight), perform the computation,
    then convert to the out_layout.


    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    weight : relax.Expr
        The weight expressions.

    strides : Union[PrimExprLike, Tuple[PrimExprLike]]
        The strides of convolution. It is required to have length either 1 or 2.

    padding : Union[PrimExprLike, Tuple[PrimExprLike]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1, 2 or 4.

    dilation : Union[PrimExprLike, Tuple[PrimExprLike]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1 or 2.

    data_layout : str
        Layout of the input.

    kernel_layout : str
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    out_dtype : Optional[Union[str, DataType]]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(strides, (int, PrimExpr)):
        strides = (strides, strides)
    if isinstance(dilation, (int, PrimExpr)):
        dilation = (dilation, dilation)
    if isinstance(padding, (int, PrimExpr)):
        padding = (padding, padding, padding, padding)

    return _ffi_api.conv2d(  # type: ignore
        data,
        weight,
        strides,
        padding,
        dilation,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def max_pool2d(
    data: Expr,
    pool_size: Union[PrimExprLike, Tuple[PrimExprLike]] = (1, 1),
    strides: Union[PrimExprLike, Tuple[PrimExprLike]] = (1, 1),
    padding: Union[PrimExprLike, Tuple[PrimExprLike]] = (0, 0),
    dilation: Union[PrimExprLike, Tuple[PrimExprLike]] = (1, 1),
    layout: str = "NCHW",
    out_layout: Optional[str] = None,
) -> Expr:
    r"""2D maximum pooling operator.

    This operator takes data as input and does 2D max value calculation
    with in pool_size sized window by striding defined by stride


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w) and pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \max_{m=0, \ldots, kh-1} \max_{n=0, \ldots, kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    pool_size : Union[PrimExprLike, Tuple[PrimExprLike]]
        The size of window for pooling. It is required to have length either 1 or 2.

    strides : Union[PrimExprLike, Tuple[PrimExprLike]]
        The strides of pooling. It is required to have length either 1 or 2.

    padding : Union[PrimExprLike, Tuple[PrimExprLike]]
        The padding for pooling. It is required to have length either 1, 2 or 4.

    dilation : Union[PrimExprLike, Tuple[PrimExprLike]]
        The dilation of pooling. It is required to have length either 1 or 2.

    layout : str
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(pool_size, (int, PrimExpr)):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, (int, PrimExpr)):
        strides = (strides, strides)
    if isinstance(dilation, (int, PrimExpr)):
        dilation = (dilation, dilation)
    if isinstance(padding, (int, PrimExpr)):
        padding = (padding, padding, padding, padding)

    return _ffi_api.max_pool2d(  # type: ignore
        data, pool_size, strides, padding, dilation, layout, out_layout
    )


def adaptive_avg_pool2d(
    data: Expr,
    output_size: Optional[Union[PrimExprLike, Tuple[PrimExprLike]]] = None,
    layout: str = "NCHW",
    out_layout: Optional[str] = None,
) -> Expr:
    r"""2D adaptive average pooling operator. This operator is experimental.

    This operator takes data as input and does 2D average value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size) for any input (NCHW).

        If a tuple of integers (height, width) are provided for output_size,
        the output size is (N x C x height x width) for any input (NCHW).

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    output_size : Optional[Union[PrimExprLike, Tuple[PrimExprLike]]]
        Output height and width.
        If not specified, it will be the same as the input height and width.
        If specified, it is required to have length either 1 or 2.

    layout : str
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(output_size, (int, PrimExpr)):
        output_size = (output_size, output_size)
    return _ffi_api.adaptive_avg_pool2d(data, output_size, layout, out_layout)  # type: ignore
