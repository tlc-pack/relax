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
# pylint: disable=abstract-method,invalid-name,missing-class-docstring,missing-function-docstring,missing-module-docstring,unused-argument
import logging
from typing import Callable, Dict, List, Optional, Union

import tvm
from tvm import te, tir, topi, relax
from tvm.relax import struct_info
from tvm.ir.module import IRModule

from ..analysis import remove_all_unused
from ..expr import Call, Constant, Expr, Function, ShapeExpr, Tuple, TupleGetItem, Var
from ..expr_functor import mutator, PyExprMutator
from ..block_builder import BlockBuilder


##################### Commons #####################

# The function type of a TE function, which accepts TE Tensors and
# other attributes, and returns the output TE Tensor.
TEFunc = Callable[..., te.Tensor]

# The function type of a legalization function, which takes a
# BlockBuilder and the Call to be legalized, and outputs the legalization
# result Expr.
LegalizeFunc = Callable[[BlockBuilder, Call], Expr]


def has_known_shape_value(sinfo: struct_info.StructInfo) -> bool:
    """Check if a given Tensor/Shape/TupleStructInfo contains
    shapes whose values are all known.

    Parameters
    ----------
    sinfo : struct_info.StructInfo
        The struct info to be checked.

    Returns
    -------
    ret : bool
        A boolean indicating if the given struct info contains shape
        values that are all known.
    """
    if isinstance(sinfo, struct_info.TensorStructInfo):
        return isinstance(sinfo.shape, ShapeExpr)
    elif isinstance(sinfo, struct_info.ShapeStructInfo):
        return sinfo.values is not None
    elif isinstance(sinfo, struct_info.TupleStructInfo):
        return all([has_known_shape_value(field_sinfo) for field_sinfo in sinfo.fields])
    elif isinstance(sinfo, struct_info.PrimStructInfo):
        return True
    else:
        return False


def try_convert_to_scalar_const(expr: Expr) -> Union[Expr, bool, float, int]:
    """Check if the input Expr is a scalar constant.
    If it is, return its plain value.
    If it is not, return the input expr.

    Parameters
    ----------
    expr : Expr
        The expr to be checked and converted.

    Returns
    --–----
    ret : Union[Expr, bool, float, int]
        Return a Python native value (int/float/bool) if the given
        expr is a scalar constant. Or return the input itself
        if it is not.
    """
    if isinstance(expr, Constant) and expr.struct_info.ndim == 0:
        return expr.data.numpy()[()].item()
    else:
        return expr


def _unary(te_func: TEFunc) -> LegalizeFunc:
    def unary_call_te(bb: BlockBuilder, call: Call) -> Expr:
        return bb.call_te(te_func, call.args[0])

    return unary_call_te


def _binary(te_func: TEFunc) -> LegalizeFunc:
    def binary_call_te(bb: BlockBuilder, call: Call) -> Expr:
        # To simplify the created PrimFunc, we first check if arg1 is a constant scalar.
        # If it is not, we then check if arg0 is a constant scalar.
        arg0 = call.args[0]
        arg1 = try_convert_to_scalar_const(call.args[1])
        if isinstance(arg1, Expr):  # type: ignore
            arg0 = try_convert_to_scalar_const(arg0)
        return bb.call_te(te_func, arg0, arg1)

    return binary_call_te


##################### Creation #####################


def _full(is_like: bool, fill_value: Optional[float], primfunc_name: str) -> LegalizeFunc:
    def full_call_te(bb: BlockBuilder, call: Call) -> Expr:
        _fill_value = (
            try_convert_to_scalar_const(call.args[1]) if fill_value is None else fill_value
        )

        return bb.call_te(
            topi.full,
            call.args[0].struct_info.shape if is_like else call.args[0],
            call.struct_info.dtype,
            _fill_value,
            primfunc_name_hint=primfunc_name,
        )

    return full_call_te


def _tril_triu(is_upper: bool, primfunc_name: str) -> LegalizeFunc:
    def tril_triu_call_te(bb: BlockBuilder, call: Call) -> Expr:
        return bb.call_te(
            topi.trilu,
            call.args[0],
            tir.const(call.attrs.k, "int32"),
            upper=is_upper,
            primfunc_name_hint=primfunc_name,
        )

    return tril_triu_call_te


##################### Datatype #####################


def _astype(bb: BlockBuilder, call: Call) -> Expr:
    arg = try_convert_to_scalar_const(call.args[0])
    if isinstance(arg, Expr):  # type: ignore
        return bb.call_te(topi.cast, arg, call.attrs.dtype)
    else:
        return relax.const(arg, call.attrs.dtype)


##################### Indexing #####################


def _take(bb: BlockBuilder, call: Call) -> Expr:
    # Currently Relax `take` operator doesn't provide the mode choices and
    # requires input indices to be in range.
    # We use fast mode, which leads to runtime error whenever some index is
    # out of bound.
    return bb.call_te(topi.take, call.args[0], call.args[1], call.attrs.axis, mode="fast")


def _strided_slice(bb: BlockBuilder, call: Call) -> Expr:
    if not all(
        [
            isinstance(call.args[0].struct_info.shape.values[i.value], tir.IntImm)
            for i in call.attrs.axes
        ]
    ):
        logging.info(
            "Cases where an axis with symbolic length is sliced are not able "
            "to be legalized through TOPI"
        )
        return call

    return bb.call_te(
        topi.strided_slice,
        call.args[0],
        call.attrs.begin,
        call.attrs.end,
        call.attrs.strides,
        call.attrs.axes,
        slice_mode="end",
    )


##################### Linear algebra #####################


def _matmul(bb: BlockBuilder, call: Call) -> Expr:
    def te_matmul(a: te.Tensor, b: te.Tensor) -> te.Tensor:
        a_shape = list(a.shape)
        b_shape = list(b.shape)
        a_prepended = False
        b_appended = False
        if len(a_shape) == 1:
            a_prepended = True
            a_shape.insert(0, 1)
        if len(b_shape) == 1:
            b_appended = True
            b_shape.append(1)

        is_a_larger = len(a_shape) > len(b_shape)
        offset = len(a_shape) - len(b_shape) if is_a_larger else len(b_shape) - len(a_shape)

        a_relax = relax.Var("a", relax.TensorStructInfo(a.shape))
        b_relax = relax.Var("b", relax.TensorStructInfo(b.shape))
        f_infer_sinfo = call.op.get_attr("FInferStructInfo")
        output_shape = f_infer_sinfo(relax.op.matmul(a_relax, b_relax), bb).shape

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
                    a_dim = a_shape[i if is_a_larger else i - offset]
                    b_dim = b_shape[i if not is_a_larger else i - offset]
                    a_dim_is_one = isinstance(a_dim, tir.IntImm) and a_dim == 1
                    b_dim_is_one = isinstance(b_dim, tir.IntImm) and b_dim == 1
                    a_indices.append(0 if a_dim_is_one else idx_spatial[i])
                    b_indices.append(0 if b_dim_is_one else idx_spatial[i])
                if not a_prepended:
                    a_indices.append(idx_spatial[-2 + b_appended])
                a_indices.append(idx_reduce)
                b_indices.append(idx_reduce)
                if not b_appended:
                    b_indices.append(idx_spatial[-1])

                dtype = call.attrs.out_dtype
                if dtype != "":
                    return a(*a_indices).astype(dtype) * b(*b_indices).astype(dtype)
                else:
                    return a(*a_indices) * b(*b_indices)

            return te.sum(multiply_compute(k), axis=k)

        return te.compute(
            output_shape,
            lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
            name="matmul",
        )

    return bb.call_te(te_matmul, call.args[0], call.args[1], primfunc_name_hint="matmul")


##################### Manipulation #####################


def _reshape(
    te_func: TEFunc, primfunc_name: str, is_collapse_sum_like: bool = False
) -> LegalizeFunc:
    def reshape_call_te(bb: BlockBuilder, call: Call):
        tgt_shape = call.args[1].struct_info.shape if is_collapse_sum_like else call.args[1]
        return bb.call_te(te_func, call.args[0], tgt_shape, primfunc_name_hint=primfunc_name)

    return reshape_call_te


def _concat(bb: BlockBuilder, call: Call) -> Expr:
    t = call.args[0]
    n_field = len(t.struct_info.fields)
    while isinstance(t, Var):
        binding = bb.lookup_binding(t)
        if not isinstance(binding, (Tuple, Var)):
            break
        t = binding

    assert isinstance(t, (Tuple, Var))
    fields = (
        t.fields if isinstance(t, Tuple) else [bb.emit(TupleGetItem(t, i)) for i in range(n_field)]
    )
    return bb.call_te(
        topi.concatenate, fields, None if call.attrs.axis is None else call.attrs.axis.value
    )


def _expand_dims(bb: BlockBuilder, call: Call) -> Expr:
    def te_expand_dims(data, axis):
        data_relax = relax.Var("data", relax.TensorStructInfo(data.shape))
        f_infer_sinfo = call.op.get_attr("FInferStructInfo")
        output_shape = f_infer_sinfo(relax.op.expand_dims(data_relax, axis), bb).shape
        output_ndim = len(output_shape)

        data_dims = []
        for i in range(output_ndim):
            if i not in axis and (i - output_ndim) not in axis:
                data_dims.append(i)
        return te.compute(
            output_shape,
            lambda *idx: data(*[idx[dim] for dim in data_dims]),
            name="expand_dims",
        )

    return bb.call_te(
        te_expand_dims, call.args[0], call.attrs.axis, primfunc_name_hint="expand_dims"
    )


def _flatten(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.reshape, call.args[0], call.struct_info.shape.values)


def _permute_dims(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.transpose, call.args[0], call.attrs.axes)


def _split(bb: BlockBuilder, call: Call) -> Expr:
    if isinstance(call.attrs.indices_or_sections, tir.IntImm):
        indices_or_sections = call.attrs.indices_or_sections.value
        modulo = tvm.arith.Analyzer().simplify(
            call.args[0].struct_info.shape.values[call.attrs.axis] % indices_or_sections
        )
        if modulo != 0:
            logging.info(
                "Split cannot be legalized by TOPI when the axis being split has "
                "length that not divisible by the input number of section."
            )
            return call
    else:
        indices_or_sections = call.attrs.indices_or_sections
    return bb.call_te(topi.split, call.args[0], indices_or_sections, call.attrs.axis)


def _squeeze(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.squeeze, call.args[0], call.attrs.axis)


##################### Search #####################


def _where(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.where, call.args[0], call.args[1], call.args[2])


##################### Statistical #####################


def _statistical(te_func: TEFunc) -> LegalizeFunc:
    def statistical_call_te(bb: BlockBuilder, call: Call) -> Expr:
        return bb.call_te(te_func, call.args[0], call.attrs.axis, call.attrs.keepdims)

    return statistical_call_te


def _compute_shape_prod(x: te.Tensor, axis: List[tir.IntImm]) -> tir.PrimExpr:
    shape_prod = tir.const(1, "int32")
    axes = [_axis.value for _axis in axis] if axis is not None else range(0, len(x.shape))
    for dim in axes:
        shape_prod = shape_prod * x.shape[dim]
    return shape_prod


def _te_mean(x: te.Tensor, axis: List[tir.IntImm], keepdims: bool) -> te.Tensor:
    shape_prod = _compute_shape_prod(x, axis)
    res_sum = topi.sum(x, axis, keepdims)
    return topi.divide(res_sum, shape_prod)


def _te_variance(x: te.Tensor, axis: List[tir.IntImm], keepdims: bool) -> te.Tensor:
    dev = x - _te_mean(x, axis, keepdims)
    return _te_mean(dev * dev, axis, keepdims)


def _mean(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        _te_mean, call.args[0], call.attrs.axis, call.attrs.keepdims, primfunc_name_hint="mean"
    )


def _std(bb: BlockBuilder, call: Call) -> Expr:
    def te_std(x: te.Tensor, axis: List[tir.IntImm], keepdims: bool) -> te.Tensor:
        return topi.sqrt(_te_variance(x, axis, keepdims))

    return bb.call_te(
        te_std, call.args[0], call.attrs.axis, call.attrs.keepdims, primfunc_name_hint="std"
    )


def _variance(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        _te_variance,
        call.args[0],
        call.attrs.axis,
        call.attrs.keepdims,
        primfunc_name_hint="variance",
    )


##################### Neural network #####################


def _nn_conv2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if len(call.attrs.data_layout) != 4 or len(call.attrs.kernel_layout) != 4:
        logging.info(
            "Conv2D where data layout or kernel layout have channel chunk "
            "cannot be legalized by TOPI at this moment."
        )
        return call
    if call.attrs.groups != 1:
        data_layout = tir.layout(call.attrs.data_layout)
        kernel_layout = tir.layout(call.attrs.kernel_layout)
        ic = call.args[0].struct_info.shape.values[data_layout.index_of("C")]
        oc = call.args[1].struct_info.shape.values[kernel_layout.index_of("O")]
        if not isinstance(ic, tir.IntImm) or not isinstance(oc, tir.IntImm):
            logging.info(
                "Conv2D where number of groups is more than one and input or output "
                "channel size is symbolic cannot be legalized by TOPI at this moment."
            )
            return call

    return bb.call_te(
        topi.nn.conv,
        inp=call.args[0],
        filt=call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        groups=call.attrs.groups,
        data_layout=call.attrs.data_layout,
        kernel_layout=call.attrs.kernel_layout,
        out_dtype=call.attrs.out_dtype if call.attrs.out_dtype != "" else None,
        primfunc_name_hint="conv2d",
    )


def _nn_max_pool2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI max_pool2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool2d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="max",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        primfunc_name_hint="max_pool2d",
    )


def _nn_adaptive_max_pool2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI adaptive_max_pool2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    def te_adaptive_avg_pool2d(data, output_size, layout_str):
        if output_size is None:
            layout = tir.layout(layout_str)
            idx_H = layout.index_of("H")
            idx_W = layout.index_of("W")
            assert idx_H != -1 and idx_W != -1
            output_size = (data.shape[idx_H], data.shape[idx_W])

        return topi.nn.adaptive_pool(data, output_size, "avg", layout_str)

    return bb.call_te(
        te_adaptive_avg_pool2d,
        call.args[0],
        call.attrs.output_size,
        call.attrs.layout,
        primfunc_name_hint="adaptive_avg_pool2d",
    )


def _nn_relu(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.nn.relu, call.args[0])


def _nn_gelu(bb: BlockBuilder, call: Call) -> Expr:
    def gelu(x: te.Tensor):
        dtype = x.dtype
        return x * (
            tir.const(0.5, dtype)
            + topi.erf(x * tir.const(0.5**0.5, dtype)) * tir.const(0.5, dtype)
        )

    return bb.call_te(gelu, call.args[0], primfunc_name_hint="gelu")


def _nn_silu(bb: BlockBuilder, call: Call) -> Expr:
    def te_silu(x: te.Tensor):
        return topi.multiply(x, topi.sigmoid(x))

    return bb.call_te(te_silu, call.args[0], primfunc_name_hint="silu")


def _nn_softmax(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.nn.softmax, call.args[0], call.attrs.axis)


def _nn_log_softmax(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.nn.log_softmax, call.args[0], call.attrs.axis)


def _nn_cross_entropy_without_logits(bb: BlockBuilder, call: Call):
    def te_cross_entropy_without_logits(x, y):
        if len(x.shape) > 1:
            return -topi.sum(topi.log(x) * y) / x.shape[0]
        return -topi.sum(topi.log(x) * y)

    return bb.call_te(
        te_cross_entropy_without_logits,
        call.args[0],
        call.args[1],
        primfunc_name_hint="cross_entropy_without_logits",
    )


def _nn_cross_entropy_with_logits(bb: BlockBuilder, call: Call):
    def te_cross_entropy_with_logits(x, y):
        if len(x.shape) > 1:
            return -topi.sum(x * y) / x.shape[0]
        return -topi.sum(x * y)

    return bb.call_te(
        te_cross_entropy_with_logits,
        call.args[0],
        call.args[1],
        primfunc_name_hint="cross_entropy_with_logits",
    )


def _nn_batch_norm(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.nn.batch_norm,
        data=call.args[0],
        gamma=call.args[1],
        beta=call.args[2],
        moving_mean=call.args[3],
        moving_var=call.args[4],
        axis=call.attrs.axis,
        epsilon=call.attrs.epsilon,
        center=call.attrs.center,
        scale=call.attrs.scale,
    )


def _nn_layer_norm(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.nn.layer_norm,
        call.args[0],
        call.args[1],
        call.args[2],
        axis=call.attrs.axes,
        epsilon=call.attrs.epsilon,
    )


def _nn_dropout(bb: BlockBuilder, call: Call) -> Expr:
    logging.info("Dropout is handled by frontend translator at this moment and is not legalized.")
    return call


def _nn_nll_loss(bb: BlockBuilder, call: Call) -> Expr:
    if len(call.args) == 2:
        # TODO(relax-team): handle optional arugment weight of NLLLoss
        logging.info(
            "Can not legalize it now, because don't know how to set "
            "the default value of the optional argument 'weight' of NLLLoss."
        )
        return call
    return bb.call_te(
        topi.nn.nll_loss,
        call.args[0],
        call.args[1],
        call.args[2],
        reduction=call.attrs.reduction,
        ignore_index=call.attrs.ignore_index,
    )


##################### Image #####################


def _image_resize2d(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.image.resize2d,
        call.args[0],
        roi=call.attrs.roi,
        size=call.args[1],
        layout=call.attrs.layout,
        method=call.attrs.method,
        coordinate_transformation_mode=call.attrs.coordinate_transformation_mode,
        rounding_method=call.attrs.rounding_method,
        bicubic_alpha=call.attrs.cubic_alpha,
        bicubic_exclude=call.attrs.cubic_exclude,
        extrapolation_value=call.attrs.extrapolation_value,
    )


##################### Common #####################


def _call_topi(te_func: TEFunc) -> LegalizeFunc:
    return lambda bb, call: bb.call_te(te_func, *call.args)


##########################################################


# Todo(relax-team): Introduce cumsum for GPT-2
# def _cumsum(bb: BlockBuilder, call: Call):
#     return bb.call_te(topi.cumsum, args[0], attrs.axis)


DEFAULT_OP_LEGALIZE_MAP: Dict[str, LegalizeFunc] = {
    # Arithmetic and comparison
    "relax.abs": _unary(topi.abs),
    "relax.cos": _unary(topi.cos),
    "relax.log": _unary(topi.log),
    "relax.exp": _unary(topi.exp),
    "relax.negative": _unary(topi.negative),
    "relax.sigmoid": _unary(topi.sigmoid),
    "relax.sin": _unary(topi.sin),
    "relax.sqrt": _unary(topi.sqrt),
    "relax.tanh": _unary(topi.tanh),
    "relax.clip": _call_topi(topi.clip),
    "relax.add": _binary(topi.add),
    "relax.divide": _binary(topi.divide),
    "relax.floor_divide": _binary(topi.floor_divide),
    "relax.multiply": _binary(topi.multiply),
    "relax.subtract": _binary(topi.subtract),
    "relax.equal": _binary(topi.equal),
    "relax.greater": _binary(topi.greater),
    "relax.greater_equal": _binary(topi.greater_equal),
    "relax.less": _binary(topi.less),
    "relax.less_equal": _binary(topi.less_equal),
    "relax.not_equal": _binary(topi.not_equal),
    # Creation
    "relax.full": _full(is_like=False, fill_value=None, primfunc_name="full"),
    "relax.full_like": _full(is_like=True, fill_value=None, primfunc_name="full"),
    "relax.ones": _full(is_like=False, fill_value=1.0, primfunc_name="ones"),
    "relax.ones_like": _full(is_like=True, fill_value=1.0, primfunc_name="ones"),
    "relax.zeros": _full(is_like=False, fill_value=0.0, primfunc_name="zeros"),
    "relax.zeros_like": _full(is_like=True, fill_value=0.0, primfunc_name="zeros"),
    "relax.tril": _tril_triu(is_upper=False, primfunc_name="tril"),
    "relax.triu": _tril_triu(is_upper=True, primfunc_name="triu"),
    # Datatype
    "relax.astype": _astype,
    # Indexing
    "relax.take": _take,
    "relax.strided_slice": _strided_slice,
    # Linear algebra
    "relax.matmul": _matmul,
    # Manipulation
    "relax.broadcast_to": _reshape(topi.broadcast_to, "broadcast_to"),
    "relax.concat": _concat,
    "relax.expand_dims": _expand_dims,
    "relax.flatten": _flatten,
    "relax.permute_dims": _permute_dims,
    "relax.reshape": _reshape(topi.reshape, "reshape"),
    "relax.split": _split,
    "relax.squeeze": _squeeze,
    # TODO(relax-team): collapse_sum support symbolic shape
    "relax.collapse_sum_like": _reshape(
        topi.collapse_sum, "collapse_sum", is_collapse_sum_like=True
    ),
    "relax.collapse_sum_to": _reshape(topi.collapse_sum, "collapse_sum"),
    # Search
    "relax.where": _where,
    # Statistical
    "relax.max": _statistical(topi.max),
    "relax.mean": _mean,
    "relax.min": _statistical(topi.min),
    "relax.prod": _statistical(topi.prod),
    "relax.std": _std,
    "relax.sum": _statistical(topi.sum),
    "relax.variance": _variance,
    # Neural network
    "relax.nn.conv2d": _nn_conv2d,
    "relax.nn.max_pool2d": _nn_max_pool2d,
    "relax.nn.adaptive_avg_pool2d": _nn_adaptive_max_pool2d,
    "relax.nn.relu": _nn_relu,
    "relax.nn.gelu": _nn_gelu,
    "relax.nn.silu": _nn_silu,
    "relax.nn.softmax": _nn_softmax,
    "relax.nn.log_softmax": _nn_log_softmax,
    "relax.nn.cross_entropy_without_logits": _nn_cross_entropy_without_logits,
    "relax.nn.cross_entropy_with_logits": _nn_cross_entropy_with_logits,
    "relax.nn.batch_norm": _nn_batch_norm,
    "relax.nn.layer_norm": _nn_layer_norm,
    "relax.nn.dropout": _nn_dropout,
    "relax.nn.nll_loss": _nn_nll_loss,
    # Image
    "relax.image.resize2d": _image_resize2d,
    # Todo(relax-team): Introduce cumsum for GPT-2
    # "relax.cumsum": _cumsum,
}


@tvm.transform.module_pass(opt_level=0, name="LegalizeOps")
class LegalizeOps:
    """Legalize high-level operator calls in Relax functions to call_tir
    with corresponding low-level TIR PrimFuncs.

    For each high-level operator, we register the way of legalizing it as a
    function, which takes a context BlockBuilder and the Call being legalized
    as input, and returns the legalized call. Here the input BlockBuilder is
    mainly used for adding the PrimFunc created by call_te into the context
    IRModule.

    The legalization function for each operator is registered in a map,
    where the operator name is the key. The default legalization functions
    are in the map `DEFAULT_OP_LEGALIZE_MAP`.

    This pass provides customizability for users to use their own legalization
    function for operators. The pass takes an optional customized map,
    which has the same key/value type as the default map (see `LegalizeFunc`),
    from users. When an operator is contained in both the default map and the
    customized map, the default legalization function will be overridden, and
    only the customized one will be used.

    Parameters
    ----------
    customize_legalize_map : Optional[Dict[str, LegalizeFunc]]
        The customized operator legalization function map.
        If not specified, it will be a fresh empty dict.
        If an op has legalization function in both the default map and the
        customized map, the customized function will override the default
        one.

    Examples
    --------
    The following code shows how to use this pass:

    .. code-block:: python

        # Define the pass input IRModule
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                z: R.Tensor((2, 3), "float32") = R.add(x, y)
                r: R.Tensor((2, 3), "float32") = R.multiply(y, z)
                return r

        # Define the customized legalization function for "relax.add"
        def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
            from tvm import topi
            return bb.call_te(topi.add, call.args[1], call.args[0])

        # Apply the pass with the customized function to the module.
        mod = LegalizeOps({"relax.add": customize_legalize_add})(Module)

    Print out the result by `mod.show()`, we can see the IRModule after
    legalization becomes

    .. code-block:: python

        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                z = R.call_tir(add, (y, x), (2, 3), dtype="float32")
                r = R.call_tir(multiply, (y, z), (2, 3), dtype="float32")
                return r

            @T.prim_func
            def add(
                A: T.Buffer[(2, 3), "float32"],
                B: T.Buffer[(2, 3), "float32"],
                T_add: T.Buffer[(2, 3), "float32"],
            ):
                T.func_attr({"tir.noalias": True})
                for ax0, ax1 in T.grid(2, 3):
                    with T.block("T_add"):
                        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                        T.writes(T_add[v_ax0, v_ax1])
                        T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

            @T.prim_func
            def multiply(
                A: T.Buffer[(2, 3), "float32"],
                B: T.Buffer[(2, 3), "float32"],
                T_multiply: T.Buffer[(2, 3), "float32"],
            ):
                T.func_attr({"tir.noalias": True})
                for ax0, ax1 in T.grid(2, 3):
                    with T.block("T_multiply"):
                        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                        T.writes(T_multiply[v_ax0, v_ax1])
                        T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[v_ax0, v_ax1]
    """

    def __init__(self, customize_legalize_map: Optional[Dict[str, LegalizeFunc]] = None):
        if customize_legalize_map is None:
            self.customize_legalize_map = dict()
        else:
            self.customize_legalize_map = customize_legalize_map

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        @mutator
        class OperatorLegalizer(PyExprMutator):
            def __init__(self, mod: IRModule, customize_legalize_map: Dict[str, LegalizeFunc]):
                super().__init__(mod)
                self.mod = mod
                self.legalize_map = DEFAULT_OP_LEGALIZE_MAP.copy()
                for name, func in customize_legalize_map.items():
                    self.legalize_map[name] = func

            def _convert_op(self, call: Call) -> Expr:
                if call.op.name in self.legalize_map:
                    # We only transform the op calls with known shape values
                    if not all(
                        [has_known_shape_value(arg.struct_info) for arg in call.args]
                    ) or not has_known_shape_value(call.struct_info):
                        return call
                    return self.legalize_map[call.op.name](self.builder_, call)
                if call.op.name != "relax.call_tir":
                    logging.warning("No legalization func for %s is found.", call.op.name)
                return call

            def transform(self) -> IRModule:
                for global_var, func in self.mod.functions.items():
                    if not isinstance(func, Function):
                        continue
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(global_var, updated_func)

                return self.builder_.get()

            def visit_call_(self, call):  # pylint: disable=arguments-differ
                call = self.visit_expr_post_order(call)
                if not isinstance(call.op, tir.op.Op):
                    return call
                return self._convert_op(call)

        return OperatorLegalizer(mod, self.customize_legalize_map).transform()
