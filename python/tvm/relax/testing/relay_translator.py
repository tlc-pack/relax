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
"""Relay to Relax translator."""

from __future__ import annotations
from typing import Dict
import tvm
from tvm.relay import Call, TupleGetItem
from tvm.relax.testing.topi import *
from tvm import relax, relay, topi, te
from tvm.relax.testing import nn
from tvm.relay.op.transform import broadcast_to, unique
import os
from tvm.script import relax as R, tir as T


# load a relay program in text format to an IRModule
def load_text(file_path: str) -> tvm.IRModule:
    if os.path.isfile(file_path):
        text_file = open(file_path, "r")
        data = text_file.read()
        text_file.close()
        mod = tvm.parser.fromtext(data)
        return mod
    else:
        raise RuntimeError(f"File at path {file_path} does not exist")


class RelayOpConverter(object):
    """A helper class for holding Relay op converters."""

    @classmethod
    def get_converter(cls):
        """Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, "_impl"):
            return getattr(cls, "_impl")
        raise tvm.error.OpNotImplemented(
            "Operator {} is not supported in frontend Relay.".format(cls.__name__)
        )


def AttrCvt(attrs) -> Dict:
    """Convert attributes to a dict."""
    attrs_dict = {}

    for k in attrs.keys():
        attrs_dict[k] = attrs[k]

    return attrs_dict


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


class Embedding(RelayOpConverter):
    """Operator converter for nn.embedding."""

    @classmethod
    def _impl(cls, inputs, attrs):
        def embedding(table, indices):
            oshape = list(indices.shape) + [table.shape[1]]
            return te.compute(oshape, lambda *i: table(indices(*i[:-1]), i[-1]), name="embedding")

        return nn.emit_te(embedding, *inputs)


@T.prim_func
def embedding_grad(table: T.handle, indices: T.handle, grad_in: T.handle, grad_out: T.handle):
    T.func_attr({"global_symbol": "embedding_grad"})
    m = T.var("int32")
    n = T.var("int32")
    k = T.var("int32")
    A = T.match_buffer(table, (m, n))
    B = T.match_buffer(indices, (k), "int32")
    C = T.match_buffer(grad_in, (k, n))
    D = T.match_buffer(grad_out, (m, n))

    for i in range(m):
        for j in range(n):
            D[i, j] = 0.0
    for i in range(k):
        for j in range(n):
            D[B[i], j] += C[i, j]


class EmbeddingGrad(RelayOpConverter):
    """Operator converter for nn.embedding_grad."""

    @classmethod
    def _impl(cls, inputs, attrs):
        tir_func = embedding_grad
        func_name = relax.BlockBuilder.current().get_unique_name(tir_func.__name__)
        tir_func = tir_func.with_attr("global_symbol", func_name)
        gvar = relax.GlobalVar(func_name)
        relax.BlockBuilder.current()._context_mod[gvar] = tir_func
        output_shape = inputs[0].shape
        call = relax.call_dps(output_shape, gvar, inputs)
        # dps = relax.call_dps(output_shape, embedding_grad, inputs)
        # call = relax.Call(relax.ExternFunc("test.embeeding_grad"), inputs)
        return relax.BlockBuilder.current().emit(call)


class BatchMatmul(RelayOpConverter):
    """Operator converter for nn.batch_matmul."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = AttrCvt(attrs)
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
        new_attrs = AttrCvt(attrs)
        return nn.emit_te(mean, *inputs, **new_attrs)


class Variance(RelayOpConverter):
    """Operator converter for variance."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = AttrCvt(attrs)
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
    """Operator converter for nn.bias_add."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = AttrCvt(attrs)
        return nn.emit_te(bias_add, *inputs, **new_attrs)


class Transpose(RelayOpConverter):
    """Operator converter for Transpose."""

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
        new_attrs = AttrCvt(attrs)
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
        new_attrs = AttrCvt(attrs)
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
        new_attrs = AttrCvt(attrs)
        return nn.emit_te(topi.squeeze, *inputs, **new_attrs)


# _convert_map defines maps of name to converter functor(callable)
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping (fusion), write custom topi func

# Minimal set of ops for transformer
def _get_convert_map():
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
        "nn.embedding": Embedding.get_converter(),
        "nn.dense": Dense.get_converter(),
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
        "nn.embedding_grad": EmbeddingGrad.get_converter(),
    }


def _convert_operator(op_type, inputs, attrs=None):
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
    convert_map = _get_convert_map()
    if op_type in convert_map:
        func = convert_map[op_type](inputs, attrs)
    else:
        raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(op_type))
    return func


def from_relay(func: relay.Function):
    """Convert a Relay model into an equivalent Relax Function.

    Parameters
    ----------
    func : relay.Function
        Relay function to be converted

    Returns
    -------
    mod : tvm.IRModule
        The Relax IRModule for compilation
    """
    var_map = {}
    # old tuple -> new tuple
    tuple_map = {}
    last_var = None
    params = []
    convert_map = _get_convert_map()

    def visit_func(node):
        nonlocal last_var
        if isinstance(node, relay.Var):
            var_map[node] = nn.Placeholder(
                tuple(node.type_annotation.shape), node.type_annotation.dtype, node.name_hint
            )
            params.append(var_map[node])
        elif isinstance(node, relay.Call):
            args = node.args
            new_args = []
            for arg in args:
                if arg in var_map:
                    new_args.append(var_map[arg])
                else:
                    new_args.append(arg)

            op_name = node.op.name
            if node.op.name not in convert_map:
                raise tvm.error.OpNotImplemented("Operator {} is not supported.".format(op_name))

            attrs = node.attrs
            var = _convert_operator(op_name, new_args, attrs)
            last_var = var
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
                    raise RuntimeError("field is not in var_map")
            new_tuple = relax.Tuple(new_fields)
            tuple_map[node] = new_tuple
            new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
            var_map[node] = new_tuple_var
            last_var = new_tuple_var
        elif isinstance(node, relay.TupleGetItem):
            if node.tuple_value in var_map:
                new_tuple = tuple_map[node.tuple_value]
                new_tuple_get_item_node = TupleGetItem(new_tuple, node.index)
                new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
                var_map[node] = new_tuple_get_item_var
                last_var = new_tuple_get_item_var
            else:
                raise RuntimeError("tuple is not in var_map")
        elif isinstance(node, relay.Function):
            relax.BlockBuilder.current().emit_func_output(last_var, params)

    relay.analysis.post_order_visit(func, visit_func)


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

    mod = tvm.parser.fromtext(RELAY_MODEL)

    mod = load_text("bert_16_128.txt")

    bb = relax.BlockBuilder()
    with bb.function("main"):
        from_relay(mod["main"])

    mod = bb.get()
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
