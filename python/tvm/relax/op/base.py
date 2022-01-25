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
from typing import Union, List
from . import _ffi_api
from ..expr import Expr, ShapeExpr, Tuple, Call
from ...ir import Array


def call_tir(
    shape: Union[Tuple, ShapeExpr, List[int]],
    func: Expr,
    args: Union[Tuple, List[Expr]],
    tir_vars: ShapeExpr = None,
) -> Call:
    """
    Call a destination-passing-style function and return the output.

    Parameters
    ----------
    shape: Tuple[ShapeExpr] or ShapeExpr
        The output shape. Tuple[ShapeExpr] if multiple outputs, ShapeExpr is single output.

    func : ExternFunc or PrimFunc
        The destination-passing-style function.

    args : Tuple[Expr]
        The input arguments.

    tir_vars : ShapeExpr
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
    return _ffi_api.call_tir(shape, func, args, tir_vars)
