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
# pylint: disable=unused-argument, invalid-name, no-else-return
"""Relay to Relax translator."""

from __future__ import annotations
from typing import Dict
import tvm
from tvm.ir.module import IRModule
from tvm.relax.testing.topi import mean, variance, reshape, reverse_reshape, bias_add, collapse_sum
from tvm import relax, relay, topi
from tvm.relax.testing import nn


class RelayOpConverter(object):
    """A helper class for holding Relay op converters."""

    @classmethod
    def get_converter(cls):
        """Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, "_impl"):
            return getattr(cls, "_impl")
        raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(cls.__name__))


class Unary(RelayOpConverter):
    """A helper class for unary op converters."""

    name = ""

    @classmethod
    def _impl(cls, inputs, attrs):
        assert len(inputs) == 1, "Unary op takes 1 inputs, but {} given".format(len(inputs))
        op_name = cls.name
        topi_func = getattr(topi, op_name)
        return nn.emit_te(topi_func, *inputs)


class Elemwise(RelayOpConverter):
    """A helper class for elemwise op converters."""

    name = ""

    @classmethod
    def _impl(cls, inputs, attrs):
        assert len(inputs) == 2, "Elemwise op takes 2 inputs, but {} given".format(len(inputs))
        op_name = cls.name
        topi_func = getattr(topi, op_name)
        return nn.emit_te(topi_func, *inputs)


class Add(Elemwise):
    """Operator converter for add."""

    name = "add"


class Subtract(Elemwise):
    """Operator converter for subtract."""

    name = "subtract"


class Divide(Elemwise):
    """Operator converter for divide."""

    name = "divide"


class Multiply(Elemwise):
    """Operator converter for multiply."""

    name = "multiply"


class Power(Elemwise):
    """Operator converter for power."""

    name = "power"


class Sqrt(Unary):
    """Operator converter for sqrt."""

    name = "sqrt"


class Exp(Unary):
    """Operator converter for exp."""

    name = "exp"


class Negative(Unary):
    """Operator converter for negative."""

    name = "negative"


class Erf(Unary):
    """Operator converter for erf."""

    name = "erf"


class Dense(RelayOpConverter):
    """Operator converter for dense."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.nn.dense, *inputs)


class BatchNorm(RelayOpConverter):
    """Operator converter for batch norm."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(topi.nn.batch_norm, *inputs, **new_attrs)


class Conv2D(RelayOpConverter):
    """Operator converter for conv2d."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            new_inputs.append(attrs["strides"])
            new_inputs.append(attrs["padding"])
            new_inputs.append(attrs["dilation"])
        else:
            raise RuntimeError("attrs must be provided to conv2d op.")
        return nn.emit_te(topi.nn.conv2d_nchw, *new_inputs)


class Relu(RelayOpConverter):
    """Operator converter for relu."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.nn.relu, *inputs)


class Reshape(RelayOpConverter):
    """Operator converter for dense."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            new_shape = []
            attr_newshape = attrs["newshape"]
            for dim in attr_newshape:
                new_shape.append(int(dim))

            new_inputs.append(new_shape)
        else:
            raise RuntimeError("attrs must be provided to reshape op.")
        return nn.emit_te(reshape, *new_inputs)


class BatchMatmul(RelayOpConverter):
    """Operator converter for nn.batch_matmul."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        if "out_dtype" in new_attrs:
            new_attrs["out_dtype"] = None
        if "transpose_a" in new_attrs:
            new_attrs["transpose_a"] = bool(new_attrs["transpose_a"])
        if "transpose_b" in new_attrs:
            new_attrs["transpose_b"] = bool(new_attrs["transpose_b"])
        return nn.emit_te(topi.nn.batch_matmul, *inputs, **new_attrs)


class Zeros(RelayOpConverter):
    """Operator converter for zeros."""

    @classmethod
    def _impl(cls, inputs, attrs):
        if attrs is not None:
            shape = attrs["shape"]
            dtype = attrs["dtype"]
            return nn.emit_te(topi.full, shape, dtype, 0.0)
        else:
            raise RuntimeError("attrs must be provided to zeros op.")


class Mean(RelayOpConverter):
    """Operator converter for mean."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(mean, *inputs, **new_attrs)


class Variance(RelayOpConverter):
    """Operator converter for variance."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(variance, *inputs, **new_attrs)


class ReverseReshape(RelayOpConverter):
    """Operator converter for contrib_reverse_reshape."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            new_shape = []
            attr_newshape = attrs["newshape"]
            for dim in attr_newshape:
                new_shape.append(int(dim))

            new_inputs.append(new_shape)
        else:
            raise RuntimeError("attrs must be provided to contrib_reverse_reshape op.")
        return nn.emit_te(reverse_reshape, *new_inputs)


class BiasAdd(RelayOpConverter):
    """Operator converter for bias_add."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(bias_add, *inputs, **new_attrs)


class Transpose(RelayOpConverter):
    """Operator converter for transpose."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            axes = attrs["axes"]
            new_inputs.append(axes)
        else:
            raise RuntimeError("attrs must be provided to transpose op.")
        return nn.emit_te(topi.transpose, *new_inputs)


class ExpandDims(RelayOpConverter):
    """Operator converter for expand_dims."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            axis = attrs["axis"]
            num_newaxis = attrs["num_newaxis"]
            new_inputs += [axis, num_newaxis]
        else:
            raise RuntimeError("attrs must be provided to expand_dims op.")
        return nn.emit_te(topi.expand_dims, *new_inputs)


class Cast(RelayOpConverter):
    """Operator converter for cast."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            dtype = attrs["dtype"]
            new_inputs.append(dtype)
        else:
            raise RuntimeError("attrs must be provided to cast op.")
        return nn.emit_te(topi.cast, *new_inputs)


class Softmax(RelayOpConverter):
    """Operator converter for softmax."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(topi.nn.softmax, *inputs, **new_attrs)


class Sum(RelayOpConverter):
    """Operator converter for sum."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            axis = attrs["axis"]
            keepdims = attrs["keepdims"]
            new_inputs += [axis, keepdims]

        return nn.emit_te(topi.sum, *new_inputs)


class LogSoftmax(RelayOpConverter):
    """Operator converter for log_softmax."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(topi.nn.log_softmax, *inputs, **new_attrs)


class Onehot(RelayOpConverter):
    """Operator converter for onehot."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            depth = attrs["depth"]
            axis = attrs["axis"]
            dtype = attrs["dtype"]
            new_inputs += [depth, axis, dtype]
        else:
            raise RuntimeError("attrs must be provided to one_hot op.")
        return nn.emit_te(topi.one_hot, *new_inputs)


class NotEqual(RelayOpConverter):
    """Operator converter for not_equal."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.not_equal, *inputs)


class CollapseSumTo(RelayOpConverter):
    """Operator converter for collapse_sum_to."""

    @classmethod
    def _impl(cls, inputs, attrs):
        if attrs is not None:
            shape = attrs["shape"]
            return nn.emit_te(collapse_sum, inputs[0], shape)
        else:
            raise RuntimeError("attrs must be provided to collapse_sum_to op.")


class BroadcastTo(RelayOpConverter):
    """Operator converter for broadcast_to."""

    @classmethod
    def _impl(cls, inputs, attrs):
        if attrs is not None:
            shape = attrs["shape"]
            return nn.emit_te(topi.broadcast_to, inputs[0], shape)
        else:
            raise RuntimeError("attrs must be provided to broadcast_to op.")


class CastLike(RelayOpConverter):
    """Operator converter for cast_like."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.cast, inputs[0], inputs[1].checked_type.dtype)


class Squeeze(RelayOpConverter):
    """Operator converter for squeeze."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(topi.squeeze, *inputs, **new_attrs)


class MaxPool2D(RelayOpConverter):
    """Operator converter for max_pool2d."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            new_inputs.append(attrs["pool_size"])
            new_inputs.append(attrs["strides"])
            new_inputs.append(attrs["dilation"])
            new_inputs.append(attrs["padding"])
            new_inputs.append("max")
            new_inputs.append(attrs["ceil_mode"])
            new_inputs.append(attrs["layout"])
        else:
            raise RuntimeError("attrs must be provided to max_pool2d op.")
        return nn.emit_te(topi.nn.pool2d, *new_inputs)


class GlobalAvgPool2D(RelayOpConverter):
    """Operator converter for global_avg_pool2d."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_inputs = [*inputs]
        if attrs is not None:
            new_inputs.append("avg")
            new_inputs.append(attrs["layout"])
        else:
            raise RuntimeError("attrs must be provided to global_avg_pool2d op.")
        return nn.emit_te(topi.nn.global_pool, *new_inputs)


class BatchFlatten(RelayOpConverter):
    """Operator converter for batch_flatten."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.nn.flatten, inputs[0])


# convert_map defines maps of name to converter functor(callable)
# use attr_convert if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping (fusion), write custom topi func

# Minimal set of ops for transformer
def get_convert_map():
    return {
        "add": Add.get_converter(),
        "subtract": Subtract.get_converter(),
        "divide": Divide.get_converter(),
        "multiply": Multiply.get_converter(),
        "power": Power.get_converter(),
        "sqrt": Sqrt.get_converter(),
        "exp": Exp.get_converter(),
        "erf": Erf.get_converter(),
        "negative": Negative.get_converter(),
        "reshape": Reshape.get_converter(),
        "nn.dense": Dense.get_converter(),
        "nn.batch_norm": BatchNorm.get_converter(),
        "nn.conv2d": Conv2D.get_converter(),
        "nn.relu": Relu.get_converter(),
        "nn.batch_matmul": BatchMatmul.get_converter(),
        "zeros": Zeros.get_converter(),
        "mean": Mean.get_converter(),
        "variance": Variance.get_converter(),
        "contrib_reverse_reshape": ReverseReshape.get_converter(),
        "nn.bias_add": BiasAdd.get_converter(),
        "transpose": Transpose.get_converter(),
        "expand_dims": ExpandDims.get_converter(),
        "cast": Cast.get_converter(),
        "broadcast_to": BroadcastTo.get_converter(),
        "nn.log_softmax": LogSoftmax.get_converter(),
        "nn.softmax": Softmax.get_converter(),
        "one_hot": Onehot.get_converter(),
        "sum": Sum.get_converter(),
        "not_equal": NotEqual.get_converter(),
        "collapse_sum_to": CollapseSumTo.get_converter(),
        "cast_like": CastLike.get_converter(),
        "squeeze": Squeeze.get_converter(),
        "nn.max_pool2d": MaxPool2D.get_converter(),
        "nn.global_avg_pool2d": GlobalAvgPool2D.get_converter(),
        "nn.batch_flatten": BatchFlatten.get_converter(),
    }


def convert_operator(op_type, inputs, attrs=None):
    """Convert from Relay operator to Relax operator/topi function.
    The converter must specify conversions explicitly for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_type : str
        Operator name, such as Convolution, FullyConnected
    inputs : list of Expr
        List of input inputs.
    attrs : dict
        Dict of operator attributes

    Returns
    -------
    func : tvm.relay.function.Function
        Converted relay function
    """
    convert_map = get_convert_map()
    if op_type in convert_map:
        func = convert_map[op_type](inputs, attrs)
    else:
        raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(op_type))
    return func


def attr_convert(attrs) -> Dict:
    """Convert attributes to a dict."""
    attrs_dict = {}

    for k in attrs.keys():
        attrs_dict[k] = attrs[k]

    return attrs_dict


def from_relay(func: relay.Function) -> IRModule:
    """Convert a Relay function into a Relax program.

    Parameters
    ----------
    func : relay.Function
        Relay function to be converted

    Returns
    -------
    mod : tvm.IRModule
        The Relax IRModule for compilation
    """
    # A map to store the mapping of Relay Expr to its corresponding Relax var
    var_map = {}
    # The output of the function
    output_var = None
    params = []
    convert_map = get_convert_map()

    def visit_func(node):
        nonlocal output_var
        if isinstance(node, relay.Var):
            if isinstance(node.type_annotation, relay.TensorType):
                var_map[node] = nn.Placeholder(
                    tuple(node.type_annotation.shape), node.type_annotation.dtype, node.name_hint
                )
                params.append(var_map[node])
            else:
                raise TypeError("The type of relay.Var to be translated must be of TensorType.")
        elif isinstance(node, relay.Call):
            args = node.args
            new_args = []
            for arg in args:
                if arg in var_map:
                    new_args.append(var_map[arg])

            op_name = node.op.name
            if node.op.name not in convert_map:
                raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(op_name))

            attrs = node.attrs
            var = convert_operator(op_name, new_args, attrs)
            output_var = var
            var_map[node] = var
        elif isinstance(node, relay.Constant):
            new_constant = relax.expr.Constant(node.data)
            var_map[node] = new_constant
        elif isinstance(node, relay.Tuple):
            new_fields = []
            for field in node.fields:
                if field in var_map:
                    new_fields.append(var_map[field])
                else:
                    raise RuntimeError("field is not in var_map.")
            new_tuple = relax.Tuple(new_fields)
            new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
            var_map[node] = new_tuple_var
            output_var = new_tuple_var
        elif isinstance(node, relay.TupleGetItem):
            if node.tuple_value in var_map:
                new_tuple = var_map[node.tuple_value]
                new_tuple_get_item_node = relax.TupleGetItem(new_tuple, node.index)
                new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
                var_map[node] = new_tuple_get_item_var
                output_var = new_tuple_get_item_var
            else:
                raise RuntimeError("tuple is not in var_map")
        elif isinstance(node, relay.Function):
            relax.BlockBuilder.current().emit_func_output(output_var, params)
        elif isinstance(node, tvm.ir.Op):
            pass
        else:
            raise TypeError("{} is not supported yet.".format(str(type(node))))

    bb = relax.BlockBuilder()
    with bb.function("main"):
        relay.analysis.post_order_visit(func, visit_func)

    return bb.get()


if __name__ == "__main__":
    RELAY_MODEL = """
    #[version = "0.0.5"]
    def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
        %0 = add(%a, %b);
        %1 = add(%c, %d);
        subtract(%0, %1)
    }
    """
    relay_mod = tvm.parser.fromtext(RELAY_MODEL)

    mod = from_relay(relay_mod["main"])

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
