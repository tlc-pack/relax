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
from tvm.relay.op.nn.utils import get_pad_tuple2d
from ...expr import Expr
from . import _ffi_api


def dense(data, weight, units=None, out_dtype=""):
    r"""Dense operator.
    Applies a linear transformation

    .. math::

    `Y = X * W^T`

    Parameters
    ----------
    data : Expr
        The input data to the operator,
        of shape `(d_1, d_2, ..., d_n, units_in)`.

    weight : Expr
        The weight expressions, 2-D matrix,
        of shape `(units, units_in)`.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense,
        of shape `(d_1, d_2, ..., d_n, units)`.

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.dense(data, weight, units, out_dtype)


def conv2d(
    data,
    weight,
    kernel_size,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_size[0], kernel_size[1])`
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
    data : Expr
        The input data to the operator.

    weight : Expr
        The weight expressions.

    strides : Optional[int, Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of inputs before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Number of groups for grouped convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    # TODO enforce 4-way padding in topi/nn/conv2d after #4644 merged
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)

    return _ffi_api.conv2d(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def relu(data: Expr) -> Expr:
    """Rectified linear unit.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : Expr
        The input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.relu(data)


def softmax(data: Expr, axis=-1) -> Expr:
    r"""Computes softmax.

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: Expr
        The input data to the operator.

    axis: int, optional
        The axis to sum over when computing softmax

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.softmax(data, axis)


def flatten(data: Expr) -> Expr:
    """Flatten.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : Expr
        The input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.flatten(data)


def max_pool2d(
    data: Expr,
    pool_size,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    layout="NCHW",
    out_layout="",
    ceil_mode=False,
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
    data : Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    padding = get_pad_tuple2d(padding)

    return _ffi_api.max_pool2d(
        data, pool_size, strides, padding, dilation, layout, out_layout, ceil_mode
    )
