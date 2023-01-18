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
# pylint: disable=redefined-builtin, invalid-name
"""Relax unary arithmetic operators."""
from . import _ffi_api
from ..expr import Expr


def cos(x: Expr) -> Expr:
    """Compute element-wise cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.cos(x)  # type: ignore


def exp(x: Expr) -> Expr:
    """Compute element-wise exp of data.
    Parameters
    ----------
    x : relax.Expr
        The input data
    Returns
    -------
    result : relax.Expr
        The computed result.
    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.exp(x)  # type: ignore


def log(x: Expr) -> Expr:
    """Compute element-wise natural logarithm of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.log(x)  # type: ignore


def negative(x: Expr) -> Expr:
    """Compute element-wise negative of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result
    """
    return _ffi_api.negative(x)  # type: ignore


def sigmoid(x: Expr) -> Expr:
    """Compute element-wise sigmoid of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sigmoid(x)  # type: ignore


def sin(x: Expr) -> Expr:
    """Compute element-wise sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sin(x)  # type: ignore


def sqrt(x: Expr) -> Expr:
    """Compute element-wise square root of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sqrt(x)  # type: ignore


def tanh(x: Expr) -> Expr:
    """Compute element-wise tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.tanh(x)  # type: ignore
