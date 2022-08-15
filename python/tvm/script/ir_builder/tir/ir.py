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
"""IRBuilder for TIR"""
import functools
import inspect
from numbers import Integral
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tvm.ir import Range, Type
from tvm.runtime import convert, ndarray
from tvm.tir import Broadcast as broadcast
from tvm.tir import (
    Buffer,
    BufferLoad,
    BufferRegion,
    Cast,
    CommReducer,
    IntImm,
    IterVar,
    Let,
    PrimExpr,
)
from tvm.tir import Ramp as ramp
from tvm.tir import Select, Shuffle, StringImm, Var, cast
from tvm.tir import op as _tir_op
from tvm.tir import type_annotation

from . import _ffi_api, frame


def buffer_decl(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    align=0,
    offset_factor=0,
    buffer_type="",
    axis_separators=None,
) -> Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    return _ffi_api.BufferDecl(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        "",
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def ptr(dtype, storage_scope="global"):
    return _ffi_api.Ptr(dtype, storage_scope)  # pylint: disable=no-member # type: ignore


def block(name: str = "", no_realize: bool = False) -> frame.BlockFrame:
    return _ffi_api.Block(name, no_realize)  # pylint: disable=no-member # type: ignore


def init() -> frame.BlockInitFrame:
    return _ffi_api.Init()  # pylint: disable=no-member # type: ignore


def where(predicate) -> None:
    if isinstance(predicate, bool):
        predicate = IntImm("bool", predicate)
    if isinstance(predicate, int):
        if predicate in [0, 1]:
            predicate = IntImm("bool", predicate)
        else:
            raise ValueError("Invalid value for predicate: {}".format(predicate))
    _ffi_api.Where(predicate)  # pylint: disable=no-member # type: ignore


def reads(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore
        else:
            buffer_slices = [buffer_slices[0]]  # type: ignore
    else:
        buffer_slices = list(buffer_slices)  # type: ignore
    _ffi_api.Reads(buffer_slices)  # pylint: disable=no-member # type: ignore


def writes(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore
    _ffi_api.Writes(buffer_slices)  # pylint: disable=no-member # type: ignore


def block_attr(attrs: Dict[str, Any]) -> None:
    return _ffi_api.BlockAttrs(attrs)  # pylint: disable=no-member # type: ignore


def alloc_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    return _ffi_api.AllocBuffer(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def _as_range(dom) -> Range:
    if isinstance(dom, Range):
        return dom
    if isinstance(dom, (list, tuple)):
        return Range(dom[0], dom[1])
    return Range(0, dom)


class axis:  # pylint: disable=invalid-name
    @staticmethod
    def spatial(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisSpatial(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def reduce(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisReduce(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def scan(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisScan(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def opaque(dom, binding, dtype="int32") -> IterVar:
        return _ffi_api.AxisOpaque(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def remap(kinds, bindings, dtype="int32") -> Union[List[IterVar], IterVar]:
        iter_vars = _ffi_api.AxisRemap(  # pylint: disable=no-member # type: ignore
            kinds, bindings, dtype
        )
        return iter_vars[0] if len(iter_vars) == 1 else iter_vars

    S = spatial  # pylint: disable=invalid-name
    R = reduce  # pylint: disable=invalid-name


def serial(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Serial(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def parallel(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Parallel(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def vectorized(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Vectorized(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def unroll(start, stop=None, *, annotations=None) -> frame.ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Unroll(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def thread_binding(
    start,
    stop=None,
    thread=None,
    *,
    annotations=None,
) -> frame.ForFrame:
    if thread is None:
        if not isinstance(stop, str):
            raise ValueError("Thread cannot be None for thread_binding")
        thread = stop
        stop = start
        start = 0
    elif stop is None:
        stop = start
        start = 0
    return _ffi_api.ThreadBinding(  # pylint: disable=no-member # type: ignore
        start, stop, thread, annotations
    )


def grid(*extents) -> frame.ForFrame:
    return _ffi_api.Grid(extents)  # pylint: disable=no-member # type: ignore


def prim_func() -> frame.PrimFuncFrame:
    return _ffi_api.PrimFunc()  # pylint: disable=no-member # type: ignore


def arg(name, obj):
    return _ffi_api.Arg(name, obj)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> str:
    return _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, Any]) -> None:
    return _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def func_ret(ret_type) -> Type:
    return _ffi_api.FuncRet(ret_type)  # pylint: disable=no-member # type: ignore


def match_buffer(
    param,
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="global",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> Buffer:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    return _ffi_api.MatchBuffer(  # pylint: disable=no-member # type: ignore
        param,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def preflattened_buffer(
    postflattened,
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="global",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> None:
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    _ffi_api.PreflattenedBuffer(  # pylint: disable=no-member # type: ignore
        postflattened,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def Assert(condition: PrimExpr, message: str) -> frame.AssertFrame:  # pylint: disable=invalid-name
    return _ffi_api.Assert(condition, message)  # pylint: disable=no-member # type: ignore


def let(
    v: Var,
    value: PrimExpr,
    body: PrimExpr = None,
) -> frame.LetFrame:
    if body is None:
        return _ffi_api.Let(v, value)  # pylint: disable=no-member # type: ignore
    return Let(v, value, body)


def allocate(
    extents: List[PrimExpr],
    dtype: str,
    scope: str = "",
    condition: PrimExpr = None,
    annotations=None,
) -> frame.AllocateFrame:
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.Allocate(  # pylint: disable=no-member # type: ignore
        extents, dtype, scope, condition, annotations
    )


def allocate_const(
    data: List[PrimExpr],
    dtype: str,
    extents: List[PrimExpr],
    annotations=None,
) -> frame.AllocateConstFrame:

    return _ffi_api.AllocateConst(  # pylint: disable=no-member # type: ignore
        ndarray.array(np.asarray(data, dtype)), dtype, extents, annotations
    )


def realize(
    buffer_slice: BufferRegion,
    storage_scope: str,
    condition: PrimExpr = True,
) -> frame.RealizeFrame:
    return _ffi_api.Realize(  # pylint: disable=no-member # type: ignore
        buffer_slice, storage_scope, condition
    )


def attr(node: Any, attr_key: str, value: Union[PrimExpr, str]) -> frame.AttrFrame:
    node = convert(node)
    value = convert(value)
    return _ffi_api.Attr(node, attr_key, value)  # pylint: disable=no-member # type: ignore


def While(condition: PrimExpr) -> frame.WhileFrame:  # pylint: disable=invalid-name
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.While(condition)  # pylint: disable=no-member # type: ignore


def If(condition: PrimExpr) -> frame.IfFrame:  # pylint: disable=invalid-name
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.If(condition)  # pylint: disable=no-member # type: ignore


def Then() -> frame.ThenFrame:  # pylint: disable=invalid-name
    return _ffi_api.Then()  # pylint: disable=no-member # type: ignore


def Else() -> frame.ElseFrame:  # pylint: disable=invalid-name
    return _ffi_api.Else()  # pylint: disable=no-member # type: ignore


def decl_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    align=0,
    offset_factor=0,
    buffer_type="",
    axis_separators=None,
) -> frame.DeclBufferFrame:

    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    return _ffi_api.DeclBuffer(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        "",
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def launch_thread(
    iter_var: IterVar,  # pylint: disable=redefined-outer-name
    extent: PrimExpr,
) -> frame.LaunchThreadFrame:
    return _ffi_api.LaunchThread(iter_var, extent)  # pylint: disable=no-member # type: ignore


def env_thread(thread_tag: str) -> IterVar:
    return _ffi_api.EnvThread(thread_tag)  # pylint: disable=no-member # type: ignore


def buffer_store(buffer: Buffer, value: PrimExpr, indices: List[Union[PrimExpr, slice]]) -> None:
    from tvm.arith import Analyzer  # pylint: disable=import-outside-toplevel

    expr_indices = []
    for index in indices:
        if isinstance(index, slice):
            step = 1 if index.step is None else index.step
            lanes = Analyzer().simplify((index.stop - index.start + step - 1) // step)
            if lanes == 1:
                expr_indices.append(index.start)
            else:
                expr_indices.append(ramp(index.start, step, int(lanes)))
        else:
            expr_indices.append(index)
    if isinstance(value, bool) and buffer.dtype == "bool":
        value = IntImm("bool", value)
    return _ffi_api.BufferStore(  # pylint: disable=no-member # type: ignore
        buffer, value, expr_indices
    )


def prefetch(buffer: Buffer, indices: List[PrimExpr]) -> None:
    return _ffi_api.Prefetch(buffer, indices)  # pylint: disable=no-member # type: ignore


def evaluate(value: PrimExpr) -> None:
    if isinstance(value, str):
        value = StringImm(value)
    return _ffi_api.Evaluate(value)  # pylint: disable=no-member # type: ignore


def int8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int8(expr)  # pylint: disable=no-member # type: ignore


def int16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int16(expr)  # pylint: disable=no-member # type: ignore


def int32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32(expr)  # pylint: disable=no-member # type: ignore


def int64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int64(expr)  # pylint: disable=no-member # type: ignore


def uint8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt8(expr)  # pylint: disable=no-member # type: ignore


def uint16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt16(expr)  # pylint: disable=no-member # type: ignore


def uint32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt32(expr)  # pylint: disable=no-member # type: ignore


def uint64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt64(expr)  # pylint: disable=no-member # type: ignore


def float8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float8(expr)  # pylint: disable=no-member # type: ignore


def float16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float16(expr)  # pylint: disable=no-member # type: ignore


def float32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float32(expr)  # pylint: disable=no-member # type: ignore


def float64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float64(expr)  # pylint: disable=no-member # type: ignore


def int32x4(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32x4(expr)  # pylint: disable=no-member # type: ignore


def int32x8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32x8(expr)  # pylint: disable=no-member # type: ignore


def int32x16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32x16(expr)  # pylint: disable=no-member # type: ignore


def boolean(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Boolean(expr)  # pylint: disable=no-member # type: ignore


def handle(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Handle(expr)  # pylint: disable=no-member # type: ignore


def void(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Void(expr)  # pylint: disable=no-member # type: ignore


def min(a, b):  # pylint: disable=redefined-builtin
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api.min(a, b)  # pylint: disable=no-member # type: ignore


def max(a, b):  # pylint: disable=redefined-builtin
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api.max(a, b)  # pylint: disable=no-member # type: ignore


def var(dtype, name="") -> Var:
    return Var(name, dtype)  # pylint: disable=no-member # type: ignore


def iter_var(v, dom, iter_type, thread_tag):
    iter_type = getattr(IterVar, iter_type)
    return IterVar(dom, v, iter_type, thread_tag)


def comm_reducer(combiner, identity):
    """Create a CommReducer from lambda inputs/outputs and the identities"""
    params = inspect.signature(combiner).parameters
    num_args = len(params)
    args = []
    for name, i in zip(params.keys(), identity + identity):
        if isinstance(i, int):
            args.append(Var(name, "int32"))
        else:
            args.append(Var(name, i.dtype))
    res = combiner(*args)
    if not isinstance(res, tuple):
        res = (res,)
    return CommReducer(args[: num_args // 2], args[num_args // 2 :], res, identity)


def llvm_lookup_intrinsic_id(name):
    # pylint: disable=import-outside-toplevel
    from tvm.target.codegen import llvm_lookup_intrinsic_id as f

    # pylint: enable=import-outside-toplevel
    return f(name)


def _op_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        return func(*args, **kwargs)

    return wrapped


def _dtype_forward(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"),) + args
        return func(*args, **kwargs)

    return wrapped


# pylint: disable=invalid-name

buffer_var = ptr
abs = _op_wrapper(_tir_op.abs)  # pylint: disable=redefined-builtin
fabs = abs
acos = _op_wrapper(_tir_op.acos)
acosh = _op_wrapper(_tir_op.acosh)
address_of = _op_wrapper(_tir_op.address_of)
asin = _op_wrapper(_tir_op.asin)
asinh = _op_wrapper(_tir_op.asinh)
atan = _op_wrapper(_tir_op.atan)
atan2 = _op_wrapper(_tir_op.atan2)
atanh = _op_wrapper(_tir_op.atanh)
ceil = _op_wrapper(_tir_op.ceil)
clz = _op_wrapper(_tir_op.clz)
copysign = _op_wrapper(_tir_op.copysign)
cos = _op_wrapper(_tir_op.cos)
cosh = _op_wrapper(_tir_op.cosh)
erf = _op_wrapper(_tir_op.erf)
exp = _op_wrapper(_tir_op.exp)
exp2 = _op_wrapper(_tir_op.exp2)
exp10 = _op_wrapper(_tir_op.exp10)
floor = _op_wrapper(_tir_op.floor)
ceildiv = _op_wrapper(_tir_op.ceildiv)
floordiv = _op_wrapper(_tir_op.floordiv)
floormod = _op_wrapper(_tir_op.floormod)
fmod = _op_wrapper(_tir_op.fmod)
hypot = _op_wrapper(_tir_op.hypot)
if_then_else = _op_wrapper(_tir_op.if_then_else)
infinity = _op_wrapper(_tir_op.infinity)
isfinite = _op_wrapper(_tir_op.isfinite)
isinf = _op_wrapper(_tir_op.isinf)
isnan = _op_wrapper(_tir_op.isnan)
isnullptr = _op_wrapper(_tir_op.isnullptr)
ldexp = _op_wrapper(_tir_op.ldexp)
likely = _op_wrapper(_tir_op.likely)
log = _op_wrapper(_tir_op.log)
log1p = _op_wrapper(_tir_op.log1p)
log2 = _op_wrapper(_tir_op.log2)
log10 = _op_wrapper(_tir_op.log10)
lookup_param = _op_wrapper(_tir_op.lookup_param)
max_value = _op_wrapper(_tir_op.max_value)
min_value = _op_wrapper(_tir_op.min_value)
nearbyint = _op_wrapper(_tir_op.nearbyint)
nextafter = _op_wrapper(_tir_op.nextafter)
popcount = _op_wrapper(_tir_op.popcount)
power = _op_wrapper(_tir_op.power)
q_multiply_shift = _op_wrapper(_tir_op.q_multiply_shift)
ret = _op_wrapper(_tir_op.ret)
reinterpret = _dtype_forward(_tir_op.reinterpret)
round = _op_wrapper(_tir_op.round)  # pylint: disable=redefined-builtin
rsqrt = _op_wrapper(_tir_op.rsqrt)
shift_left = _op_wrapper(_tir_op.shift_left)
shift_right = _op_wrapper(_tir_op.shift_right)
sigmoid = _op_wrapper(_tir_op.sigmoid)
sin = _op_wrapper(_tir_op.sin)
sinh = _op_wrapper(_tir_op.sinh)
sqrt = _op_wrapper(_tir_op.sqrt)
tan = _op_wrapper(_tir_op.tan)
tanh = _op_wrapper(_tir_op.tanh)
trunc = _op_wrapper(_tir_op.trunc)
truncdiv = _op_wrapper(_tir_op.truncdiv)
truncmod = _op_wrapper(_tir_op.truncmod)
tvm_access_ptr = _op_wrapper(_tir_op.tvm_access_ptr)
tvm_throw_last_error = _op_wrapper(_tir_op.tvm_throw_last_error)
tvm_stack_alloca = _op_wrapper(_tir_op.tvm_stack_alloca)
tvm_stack_make_shape = _op_wrapper(_tir_op.tvm_stack_make_shape)
tvm_stack_make_array = _op_wrapper(_tir_op.tvm_stack_make_array)
call_packed = _op_wrapper(_tir_op.call_packed)
call_cpacked = _op_wrapper(_tir_op.call_cpacked)
call_packed_lowered = _op_wrapper(_tir_op.call_packed_lowered)
call_cpacked_lowered = _op_wrapper(_tir_op.call_cpacked_lowered)
call_extern = _dtype_forward(_tir_op.call_extern)
call_intrin = _dtype_forward(_tir_op.call_intrin)
call_llvm_intrin = _dtype_forward(_tir_op.call_llvm_intrin)
call_llvm_pure_intrin = _dtype_forward(_tir_op.call_llvm_pure_intrin)
call_pure_extern = _dtype_forward(_tir_op.call_pure_extern)
tvm_access_ptr = _op_wrapper(_tir_op.tvm_access_ptr)
tvm_tuple = _op_wrapper(_tir_op.tvm_tuple)
tvm_struct_set = _op_wrapper(_tir_op.tvm_struct_set)
tvm_struct_get = _tir_op.tvm_struct_get
tvm_thread_allreduce = _op_wrapper(_tir_op.tvm_thread_allreduce)
tvm_load_matrix_sync = _op_wrapper(_tir_op.tvm_load_matrix_sync)
tvm_mma_sync = _op_wrapper(_tir_op.tvm_mma_sync)
tvm_bmma_sync = _op_wrapper(_tir_op.tvm_bmma_sync)
tvm_fill_fragment = _op_wrapper(_tir_op.tvm_fill_fragment)
tvm_store_matrix_sync = _op_wrapper(_tir_op.tvm_store_matrix_sync)
ptx_mma = _dtype_forward(_tir_op.ptx_mma)
ptx_mma_sp = _dtype_forward(_tir_op.ptx_mma_sp)
ptx_ldmatrix = _dtype_forward(_tir_op.ptx_ldmatrix)
ptx_cp_async = _dtype_forward(_tir_op.ptx_cp_async)
ptx_wait_group = _op_wrapper(_tir_op.ptx_wait_group)
ptx_commit_group = _op_wrapper(_tir_op.ptx_commit_group)
mma_store = _dtype_forward(_tir_op.mma_store)
mma_fill = _dtype_forward(_tir_op.mma_fill)
vectorlow = _dtype_forward(_tir_op.vectorlow)
vectorhigh = _dtype_forward(_tir_op.vectorhigh)
vectorcombine = _dtype_forward(_tir_op.vectorcombine)
assume = _op_wrapper(_tir_op.assume)
undef = _op_wrapper(_tir_op.undef)
tvm_call_packed = call_packed
tvm_call_cpacked = call_cpacked
tvm_call_packed_lowered = call_packed_lowered
tvm_call_cpacked_lowered = call_cpacked_lowered
TVMBackendAllocWorkspace = _op_wrapper(_tir_op.TVMBackendAllocWorkspace)
TVMBackendFreeWorkspace = _op_wrapper(_tir_op.TVMBackendFreeWorkspace)


class inline:
    def __init__(self, value) -> None:
        self.value = value
        self.i = 0

    def __iter__(self):
        def f():
            for i in self.value:
                yield inline(i)

        return f()


# pylint: enable=invalid-name


__all__ = [
    "Assert",
    "Cast",
    "Else",
    "If",
    "Let",
    "Select",
    "Shuffle",
    "TVMBackendAllocWorkspace",
    "TVMBackendFreeWorkspace",
    "Then",
    "While",
    "abs",
    "acos",
    "acosh",
    "address_of",
    "alloc_buffer",
    "allocate",
    "allocate_const",
    "arg",
    "asin",
    "asinh",
    "assume",
    "atan",
    "atan2",
    "atanh",
    "attr",
    "axis",
    "block",
    "block_attr",
    "boolean",
    "broadcast",
    "buffer_decl",
    "buffer_store",
    "buffer_var",
    "call_cpacked",
    "call_cpacked_lowered",
    "call_extern",
    "call_intrin",
    "call_llvm_intrin",
    "call_llvm_pure_intrin",
    "call_packed",
    "call_packed_lowered",
    "call_pure_extern",
    "cast",
    "ceil",
    "ceildiv",
    "clz",
    "comm_reducer",
    "copysign",
    "cos",
    "cosh",
    "env_thread",
    "erf",
    "evaluate",
    "exp",
    "exp10",
    "exp2",
    "decl_buffer",
    "fabs",
    "float16",
    "float32",
    "float64",
    "float8",
    "floor",
    "floordiv",
    "floormod",
    "fmod",
    "func_attr",
    "func_name",
    "func_ret",
    "grid",
    "handle",
    "hypot",
    "if_then_else",
    "infinity",
    "init",
    "inline",
    "int16",
    "int32",
    "int32x16",
    "int32x4",
    "int32x8",
    "int64",
    "int8",
    "isfinite",
    "isinf",
    "isnan",
    "isnullptr",
    "iter_var",
    "launch_thread",
    "ldexp",
    "let",
    "likely",
    "llvm_lookup_intrinsic_id",
    "log",
    "log10",
    "log1p",
    "log2",
    "lookup_param",
    "match_buffer",
    "max",
    "max_value",
    "min",
    "min_value",
    "mma_fill",
    "mma_store",
    "nearbyint",
    "nextafter",
    "parallel",
    "popcount",
    "power",
    "prefetch",
    "preflattened_buffer",
    "prim_func",
    "ptr",
    "ptx_commit_group",
    "ptx_cp_async",
    "ptx_ldmatrix",
    "ptx_mma",
    "ptx_mma_sp",
    "ptx_wait_group",
    "q_multiply_shift",
    "ramp",
    "reads",
    "realize",
    "reinterpret",
    "ret",
    "round",
    "rsqrt",
    "serial",
    "shift_left",
    "shift_right",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "thread_binding",
    "trunc",
    "truncdiv",
    "truncmod",
    "tvm_access_ptr",
    "tvm_bmma_sync",
    "tvm_call_cpacked",
    "tvm_call_cpacked_lowered",
    "tvm_call_packed",
    "tvm_call_packed_lowered",
    "tvm_fill_fragment",
    "tvm_load_matrix_sync",
    "tvm_mma_sync",
    "tvm_stack_alloca",
    "tvm_stack_make_array",
    "tvm_stack_make_shape",
    "tvm_store_matrix_sync",
    "tvm_struct_get",
    "tvm_struct_set",
    "tvm_thread_allreduce",
    "tvm_throw_last_error",
    "tvm_tuple",
    "type_annotation",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "undef",
    "unroll",
    "var",
    "vectorcombine",
    "vectorhigh",
    "vectorized",
    "vectorlow",
    "void",
    "where",
    "writes",
]
