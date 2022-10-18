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
from typing import List
import math

import tvm
from tvm import ir, te, topi
from tvm.ir import Attrs
from tvm.ir.module import IRModule

from ..analysis import remove_all_unused
from ..expr import Call, Expr, Function, Tuple, TupleGetItem
from ..expr_functor import mutator, PyExprMutator
from ..block_builder import BlockBuilder


def _nn_conv2d(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.nn.conv2d,
        input=args[0],
        filter=args[1],
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        data_layout=attrs.data_layout,
        kernel_layout=attrs.kernel_layout,
        out_dtype=attrs.out_dtype if attrs.out_dtype != "" else None,
    )


def _add(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.add, args[0], args[1])


def _subtract(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.subtract, args[0], args[1])


def _multiply(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.multiply, args[0], args[1])


def _divide(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.divide, args[0], args[1])


def _floor_divide(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.floor_divide, args[0], args[1])


def _sin(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.sin, args[0])


def _cos(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.cos, args[0])


def _sqrt(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.sqrt, args[0])


def _nn_relu(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.nn.relu, args[0])


def _nn_gelu(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    def gelu(x):
        return te.compute(
            x.shape,
            lambda *i: 0.5
            * x(*i)
            * (1 + te.tanh(math.sqrt(2 / math.pi) *
               (x(*i) + 0.044715 * te.power(x(*i), 3)))),
        )

    return bb.call_te(gelu, args[0])


def _nn_silu(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    sig = bb.emit_te(topi.sigmoid, args[0])
    return bb.call_te(topi.multiply, args[0], sig)


def _reshape(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.reshape, args[0], output_shape)


def _transpose(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.transpose, args[0], attrs.axes)


def _concatenate(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    n_field = len(args[0].shape_.fields)
    fields = []
    for i in range(n_field):
        fields.append(
            bb.emit(TupleGetItem(args[0], i))
            if not isinstance(args[0], Tuple)
            else args[0].fields[i]
        )
    return bb.call_te(topi.concatenate, fields, None if attrs.axis is None else attrs.axis.value)


def _expand_dims(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    output_ndim = len(output_shape)

    def expand_dims(data, axis):
        data_dims = []
        for i in range(output_ndim):
            if i not in axis and (i - output_ndim) not in axis:
                data_dims.append(i)
        return te.compute(output_shape, lambda *idx: data(*[idx[dim] for dim in data_dims]))

    return bb.call_te(expand_dims, args[0], attrs.axis)


def _cumsum(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.cumsum, args[0], attrs.axis)


def _trilu(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.trilu, args[0], tvm.tir.const(attrs.k, "int32"), attrs.is_upper)


def _cast(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.cast, args[0], attrs.dtype)


def _take(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.take, args[0], args[1], attrs.axis, attrs.batch_dims, attrs.mode)


def _full(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.full,
        args[1],
        attrs.dtype if attrs.dtype is not None else args[0].checked_type.dtype,
        args[0],
    )


def _split(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    indices_or_sections = (
        attrs.indices_or_sections.value
        if isinstance(attrs.indices_or_sections, tvm.tir.IntImm)
        else attrs.indices_or_sections
    )
    return bb.call_te(topi.split, args[0], indices_or_sections, attrs.axis)


def _broadcast_to(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.broadcast_to, args[0], args[1])


def _strided_slice(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.strided_slice,
        args[0],
        attrs.begin,
        attrs.end,
        attrs.strides,
        attrs.axes,
        attrs.slice_mode,
    )


def _nn_max_pool2d(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.nn.pool2d,
        args[0],
        kernel=attrs.pool_size,
        stride=attrs.strides,
        dilation=attrs.dilation,
        padding=attrs.padding,
        pool_type="max",
        ceil_mode=attrs.ceil_mode,
        layout=attrs.layout,
    )


def _nn_batch_norm(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.nn.batch_norm,
        data=args[0],
        gamma=args[1],
        beta=args[2],
        moving_mean=args[3],
        moving_var=args[4],
        axis=attrs.axis,
        epsilon=attrs.epsilon,
        center=attrs.center,
        scale=attrs.scale,
    )


def _nn_layer_norm(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    def layer_norm(x, gamma, beta, axis, eps):
        shape_prod = tvm.tir.const(1, "int32")
        for dim in axis:
            shape_prod = shape_prod * x.shape[dim.value]
        mean = topi.sum(x, axis=axis, keepdims=True) / shape_prod
        var = topi.sum((x - mean) * (x - mean), axis=axis,
                       keepdims=True) / shape_prod
        return gamma * ((x - mean) / topi.sqrt(var + eps)) + beta

    return bb.call_te(layer_norm, args[0], args[1], args[2], axis=attrs.axis, eps=attrs.epsilon)


def _nn_matmul(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    a = args[0]
    b = args[1]
    a_shape = list(a.shape_)
    b_shape = list(b.shape_)

    a_prepended = False
    b_appended = False
    if len(a_shape) == 1:
        a_prepended = True
        a_shape.insert(0, 1)
    if len(b_shape) == 1:
        b_appended = True
        b_shape.append(1)

    is_a_larger = len(a_shape) > len(b_shape)
    offset = len(a_shape) - \
        len(b_shape) if is_a_larger else len(b_shape) - len(a_shape)

    def matmul(a, b):
        def matmul_compute(*idx_spatial):
            k = te.reduce_axis((0, a_shape[-1]), name="k")

            def multiply_compute(idx_reduce):
                a_indices = []
                b_indices = []

                for i in range(offset):
                    if is_a_larger:
                        a_indices.append(idx_spatial[i])
                    else:
                        b_indices.append(idx_spatial[i])
                for i in range(offset, len(output_shape) - (2 - a_prepended - b_appended)):
                    a_idx = i if is_a_larger else i - offset
                    b_idx = i if not is_a_larger else i - offset
                    a_indices.append(
                        idx_spatial[i] if a_shape[a_idx] > 1 else 0)
                    b_indices.append(
                        idx_spatial[i] if b_shape[b_idx] > 1 else 0)
                if not a_prepended:
                    a_indices.append(idx_spatial[-2 + b_appended])
                a_indices.append(idx_reduce)
                b_indices.append(idx_reduce)
                if not b_appended:
                    b_indices.append(idx_spatial[-1])

                dtype = attrs.out_dtype
                if dtype != "":
                    return a(*a_indices).astype(dtype) * b(*b_indices).astype(dtype)
                else:
                    return a(*a_indices) * b(*b_indices)

            return te.sum(multiply_compute(k), axis=k)

        return te.compute(output_shape, lambda *idx: matmul_compute(*idx), name="matmul")

    return bb.call_te(matmul, a, b)


def _nn_softmax(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.nn.softmax, args[0], attrs.axis)


def _nn_flatten(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.nn.flatten, args[0])


def _nn_adaptive_max_pool2d(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.nn.adaptive_pool, args[0], attrs.output_size, pool_type="avg", layout=attrs.layout
    )


def _sum(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.sum, args[0], attrs.axis, attrs.keepdims)


def _mean(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    shape_prod = tvm.tir.const(1, "int32")
    axis = attrs.axis if attrs.axis is not None else range(
        0, len(args[0].shape))
    for dim in axis:
        shape_prod = shape_prod * args[0].shape[dim.value]
    sum_var = bb.emit_te(topi.sum, args[0], axis, attrs.keepdims)
    return bb.call_te(topi.divide, sum_var, shape_prod)


def _image_resize2d(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(
        topi.image.resize2d,
        args[0],
        roi=attrs.roi,
        size=attrs.size,
        layout=attrs.layout,
        method=attrs.method,
        coordinate_transformation_mode=attrs.coordinate_transformation_mode,
        rounding_method=attrs.rounding_method,
        bicubic_alpha=attrs.cubic_alpha,
        bicubic_exclude=attrs.cubic_exclude,
        extrapolation_value=attrs.extrapolation_value,
    )


op_legalization_map = {
    ir.Op.get("relax.nn.conv2d"): _nn_conv2d,
    ir.Op.get("relax.add"): _add,
    ir.Op.get("relax.subtract"): _subtract,
    ir.Op.get("relax.multiply"): _multiply,
    ir.Op.get("relax.divide"): _divide,
    ir.Op.get("relax.floor_divide"): _floor_divide,
    ir.Op.get("relax.sin"): _sin,
    ir.Op.get("relax.cos"): _cos,
    ir.Op.get("relax.sqrt"): _sqrt,
    ir.Op.get("relax.nn.relu"): _nn_relu,
    ir.Op.get("relax.nn.gelu"): _nn_gelu,
    ir.Op.get("relax.nn.silu"): _nn_silu,
    ir.Op.get("relax.reshape"): _reshape,
    ir.Op.get("relax.transpose"): _transpose,
    ir.Op.get("relax.concatenate"): _concatenate,
    ir.Op.get("relax.expand_dims"): _expand_dims,
    ir.Op.get("relax.cumsum"): _cumsum,
    ir.Op.get("relax.trilu"): _trilu,
    ir.Op.get("relax.cast"): _cast,
    ir.Op.get("relax.take"): _take,
    ir.Op.get("relax.full"): _full,
    ir.Op.get("relax.split"): _split,
    ir.Op.get("relax.strided_slice"): _strided_slice,
    ir.Op.get("relax.broadcast_to"): _broadcast_to,
    ir.Op.get("relax.nn.max_pool2d"): _nn_max_pool2d,
    ir.Op.get("relax.nn.batch_norm"): _nn_batch_norm,
    ir.Op.get("relax.nn.layer_norm"): _nn_layer_norm,
    ir.Op.get("relax.nn.matmul"): _nn_matmul,
    ir.Op.get("relax.nn.softmax"): _nn_softmax,
    ir.Op.get("relax.nn.flatten"): _nn_flatten,
    ir.Op.get("relax.nn.adaptive_avg_pool2d"): _nn_adaptive_max_pool2d,
    ir.Op.get("relax.sum"): _sum,
    ir.Op.get("relax.mean"): _mean,
    ir.Op.get("relax.image.resize2d"): _image_resize2d,
}


@mutator
class OperatorLegalizer(PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def _convert_op(self, call: Call) -> Call:
        if call.op in op_legalization_map:
            return op_legalization_map[call.op](self.builder_, call.args, call.attrs, call.shape_)

        if call.op.name != "relax.call_tir":
            print(call.op.name)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, Function):
                continue
            updated_func = self.visit_expr(func)
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        return self._convert_op(call)
