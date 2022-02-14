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
from typing import Dict, List
import tvm
from tvm.ir.module import IRModule
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


class Dense(RelayOpConverter):
    """Operator converter for nn.dense."""

    @classmethod
    def _impl(cls, inputs, attrs):
        return nn.emit_te(topi.nn.dense, *inputs)


class BatchNorm(RelayOpConverter):
    """Operator converter for nn.batch_norm."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(topi.nn.batch_norm, *inputs, **new_attrs)


class Conv2D(RelayOpConverter):
    """Operator converter for nn.conv2d."""

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


class Softmax(RelayOpConverter):
    """Operator converter for softmax."""

    @classmethod
    def _impl(cls, inputs, attrs):
        new_attrs = attr_convert(attrs)
        return nn.emit_te(topi.nn.softmax, *inputs, **new_attrs)


# convert_map defines maps of name to converter functor(callable)
# use attr_convert if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping (fusion), write custom topi func

# Minimal set of ops for transformer
def get_convert_map():
    return {
        "nn.dense": Dense.get_converter(),
        "nn.batch_norm": BatchNorm.get_converter(),
        "nn.conv2d": Conv2D.get_converter(),
        "nn.batch_matmul": BatchMatmul.get_converter(),
        "nn.softmax": Softmax.get_converter(),
    }


def convert_operator(op_type: str, inputs: List[relax.Expr], attrs: Dict = None):
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


def attr_convert(attrs: tvm.ir.Attrs) -> Dict:
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
            attrs = node.attrs
            compute_func = node.op.get_attr("FTVMCompute")
            if compute_func is None:
                if node.op.name not in convert_map:
                    raise tvm.error.OpNotImplemented(
                        "Operator {} is not supported.".format(op_name)
                    )
                var = convert_operator(op_name, new_args, attrs)
            else:
                name_hint = op_name.split(".")[-1]
                var = bb.emit_te(
                    compute_func, attrs, new_args, node.checked_type, primfunc_name_hint=name_hint
                )

            output_var = var
            var_map[node] = var
        elif isinstance(node, relay.Constant):
            # fill the shape and checked_type fields of the Constant
            new_constant = relay.Constant(node.data)
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
