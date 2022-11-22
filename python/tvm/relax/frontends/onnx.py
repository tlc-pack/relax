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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""ONNX: Open Neural Network Exchange frontend for Relax."""
import copy
import math
import warnings
from typing import Optional

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm import relax, topi
from tvm.relax import testing


def new_var(var_name, shape, dtype="float32"):
    return testing.nn.Parameter(shape=shape, dtype=dtype, name=var_name)


def get_type(elem_type):
    """Converts onnx integer datatype to numpy datatype"""
    # If a string was passed instead of a tensor type, it does not need
    # conversion and can be returned.
    if isinstance(elem_type, str):
        return elem_type

    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))

    try:
        from onnx import TensorProto
    except ImportError as e:
        raise ImportError("Unable to import TensorProto from onnx {}".format(e))

    # Onnx mapping converts bfloat16 to float16 because
    # numpy does not have a bfloat16 data type. However,
    # tvm has one, so we force the return type to be bfloat16
    if elem_type == int(TensorProto.BFLOAT16):
        return "bfloat16"
    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            value = tvm.tir.Var("d", "int64")
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    if info_proto.type.tensor_type.elem_type:
        dtype = get_type(info_proto.type.tensor_type.elem_type)
    else:
        dtype = None
    return name, shape, dtype, shape_name


def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)


class onnx_input(list):
    """A helper extension to list that returns None for out of bound indices."""

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.stop is None:
                stop = len(self)
            else:
                stop = item.stop
            indices = list(range(stop)[item])
            return [self[i] for i in indices]
        if isinstance(item, int):
            return list(self)[item] if item < len(self) else None
        raise TypeError("list indices must be integers or slices, not %s" % type(item).__name__)


class OnnxOpConverter(object):
    """A helper class for holding onnx op converters."""

    @classmethod
    def get_converter(cls, opset):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        versions = [int(d.replace("_impl_v", "")) for d in dir(cls) if "_impl_v" in d]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "opset version {} of {} not implemented".format(version, cls.__name__)
        )


class MatMul(OnnxOpConverter):
    """Operator converter for MatMul."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        assert len(inputs) == 2, "MatMul op takes 2 inputs, {} given".format(len(inputs))
        if len(inputs[0].shape) == 2:
            return bb.emit_te(topi.matmul, inputs[0], bb.normalize(inputs[1]))
        else:
            transpose = bb.emit_te(topi.transpose, inputs[1], [1, 0])
            expend = bb.emit_te(topi.expand_dims, transpose, 0)
            return bb.emit_te(topi.nn.batch_matmul, inputs[0], expend)


class Tanh(OnnxOpConverter):
    """Operator converter for Tanh."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        assert len(inputs) == 1, "Tanh op takes 1 input, {} given".format(len(inputs))
        return bb.emit_te(topi.tanh, inputs[0])


class Relu(OnnxOpConverter):
    """Operator converter for Relu."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        assert len(inputs) == 1, "Relu op takes 1 input, {} given".format(len(inputs))
        return bb.emit_te(topi.nn.relu, inputs[0])


class Sigmoid(OnnxOpConverter):
    """Operator converter for Sigmoid."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        assert len(inputs) == 1, "Sigmoid op takes 1 input, {} given".format(len(inputs))
        return bb.emit_te(topi.sigmoid, inputs[0])


class Gemm(OnnxOpConverter):
    """Operator converter for Gemm."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        # assert len(inputs) == 3, "Gemm op takes 3 inputs, {} given".format(len(inputs))
        a = inputs[0]
        b = inputs[1]
        c = inputs[2]
        dense = bb.emit_te(topi.nn.dense, a, b)
        if c is not None:
            return bb.emit_te(topi.add, dense, c)
        return dense


class BiasGelu(OnnxOpConverter):
    """Operator converter for BiasGelu from Microsoft onnxruntime contrib opset.

    bias_gelu(x, b) = 0.5(x + b)(1 + erf((x + b)/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        x = inputs[0]

        b = inputs[1]

        # assert len(x.shape) == 1, "BiasGelu bias term must be a 1D tensor"

        inp = bb.emit_te(topi.add, x, b)

        # Declare consts
        const_dtype = x.checked_type.dtype
        half = relax.const(0.5, const_dtype)
        one = relax.const(1.0, const_dtype)
        sqrt2 = relax.const(math.sqrt(2), const_dtype)

        # Compute gelu
        term1 = bb.emit_te(topi.multiply, half, inp)
        divide = bb.emit_te(topi.divide, inp, sqrt2)
        erf = bb.emit_te(topi.erf, divide)
        term2 = bb.emit_te(topi.add, one, erf)
        return bb.emit_te(topi.multiply, term1, term2)


def layer_norm(bb, x, eps, gamma, beta):
    """A common function to handle layer norm.

    Use LayerNormalization for the actual onnx op.
    """
    x_dtype = x.checked_type.dtype
    x_shape = [val.value for val in x.shape.values]
    num_elements = relax.const(np.prod(x_shape), dtype=x_dtype)

    # Compute Mean
    mean = bb.emit_te(topi.sum, x)
    mean = bb.emit_te(topi.divide, mean, num_elements)

    # Compute Variance
    diff = bb.emit_te(topi.subtract, x, mean)
    sq_diff = bb.emit_te(topi.multiply, diff, diff)
    var_sum = bb.emit_te(topi.sum, sq_diff, -1, True)
    var = bb.emit_te(topi.divide, var_sum, num_elements)

    # Compute Layer Normalization
    sub = bb.emit_te(topi.subtract, x, mean)
    add = bb.emit_te(topi.add, var, relax.const(eps, dtype=x_dtype))
    sqrt = bb.emit_te(topi.sqrt, add)
    output = bb.emit_te(topi.divide, sub, sqrt)
    output = bb.emit_te(topi.multiply, output, gamma)

    if beta is not None:
        output = bb.emit_te(topi.add, output, beta)

    return output


class Gather(OnnxOpConverter):
    """Operator converter for Gather."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        axis = attr.get("axis", 0)
        data = inputs[0]
        indices = inputs[1]
        return bb.emit_te(topi.take, data, indices, axis)


class Concat(OnnxOpConverter):
    """Operator converter for Concat."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        axis = attr.get("axis", 0)
        return bb.emit_te(topi.concatenate, inputs, axis)


class SkipLayerNormalization(OnnxOpConverter):
    """Operator converter for SkipLayerNormalization from Microsoft onnxruntime contrib opset.

    This layer sums the two input tensors (along with optional bias), and applies layer
    normalization.
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        data = inputs[0]
        skip = inputs[1]
        gamma = inputs[2]
        beta = inputs[3]
        bias = inputs[4]

        assert (
            beta is not None and bias is not None
        ), "SkipLayerNormalization import currently only supports required beta and bias"

        eps = attr.get("epsilon", 1e-12)

        x = bb.emit_te(topi.add, data, skip)
        if bias is not None:
            x = bb.emit_te(topi.add, x, bias)

        output = layer_norm(bb, x, eps, gamma, beta)
        return output


class EmbedLayerNormalization(OnnxOpConverter):
    """Operator converter for EmbedLayerNormalization from Microsoft onnxruntime contrib opset.

    This layer embeds the input tokens, sums them, and applies layer normalization.
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        input_ids = inputs[0]
        segment_ids = inputs[1]
        word_emb = inputs[2]
        pos_emb = inputs[3]
        segment_emb = inputs[4]
        gamma = inputs[5]
        beta = inputs[6]

        mask = inputs[7]
        pos_ids = inputs[8]

        eps = attr.get("epsilon", 1e-12)

        (batch_size, seq_len) = [val.value for val in input_ids.shape.values]

        if segment_ids:
            assert segment_emb

        if pos_ids is None:
            pos_ids = relax.const([list(range(seq_len))] * batch_size, dtype="int32")

        word_vec = bb.emit_te(topi.take, word_emb, input_ids, 0)
        segment_vec = bb.emit_te(topi.take, segment_emb, segment_ids, 0)
        pos_vec = bb.emit_te(topi.take, pos_emb, pos_ids, 0)

        vec_sum = bb.emit_te(topi.add, word_vec, pos_vec)
        if segment_ids:
            vec_sum = bb.emit_te(topi.add, vec_sum, segment_vec)

        ln = layer_norm(bb, vec_sum, eps, gamma, beta)

        mask_index = relax.const(np.zeros((batch_size,), dtype="int64"))
        if mask:
            # calculate number of words per sentence
            mask_index = bb.emit_te(topi.sum, mask, axis=1)

        # TODO(@anwang2009): onnxruntime v1.10.0 requires a third output of vec_sum
        return relax.Tuple([ln, mask_index])


class AttentionCutlass(OnnxOpConverter):
    """Operator converter for Attention from Microsoft onnxruntime contrib opset.

    This is the self-attention mechanism used in transformer models."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        num_heads = attr["num_heads"]

        assert (
            "qkv_hidden_sizes" not in attr
        ), "different hidden sizes for Q, K, V are not currently supported"
        assert "unidirectional" not in attr, "unidirectional attention not current supported"

        # (batch, seq, in_hidden)
        input_emb = inputs[0]

        # (in_hidden, 3 * out_hidden), where out_hidden = num_heads * head_size
        weight = bb.normalize(inputs[1])

        # (3 * out_hidden,)
        bias = bb.normalize(inputs[2])

        # 1. (    batch,              1,        max_seq, max_seq)
        # 2. (    batch, past_seq + seq,)
        # 3. (    batch,            seq, past_seq + seq,)
        # 4. (    batch,)
        # 5. (2 * batch,)
        # For now, we only support case 2.
        mask_index = bb.normalize(inputs[3])

        # (2, batch, num_heads, past_seq, head_size)
        past = inputs[4]

        # (batch, num_heads, seq, seq)
        extra_add = inputs[5]

        (batch_size, seq_len, _) = [val.value for val in input_emb.shape.values]

        (out_hidden_x3,) = [val.value for val in bias.shape.values]
        assert out_hidden_x3 % 3 == 0, "bias shape should be divisible by 3"
        out_hidden = out_hidden_x3 // 3
        assert (
            out_hidden % num_heads == 0
        ), "output hidden size should be divisible by number of attention heads"
        head_size = out_hidden // num_heads

        assert (
            mask_index is not None
        ), "Attention import currently only supports required mask_index"
        mask_index_shape = mask_index.shape
        assert (
            len(mask_index_shape) == 2
            and mask_index_shape[0] == batch_size
            and mask_index_shape[1] == seq_len
        ), "currently only support (batch_size, sequence_length) mask index"

        assert past is None, "past K, V state is not currently supported"
        assert extra_add is None, "extra add to QxK not currently supported"
        
        transpose = bb.emit_te(topi.transpose, weight, [1, 0])
        expend = bb.emit_te(topi.expand_dims, transpose, 0)
        batch_matmul = bb.emit_te(topi.nn.batch_matmul, input_emb, expend)
        
        qkv = bb.emit_te(topi.add, batch_matmul, bias)
        
        qkv = bb.emit(relax.call_tir(
            "FusedQKVToCxt",
            (qkv, mask_index),
            (batch_size, num_heads, seq_len, head_size),
            "float32",
        ))

        output = bb.emit_te(topi.transpose, qkv, [0, 2, 1, 3])
        output = bb.emit_te(
            topi.reshape, output, (int(output.shape[0]), int(output.shape[1]), out_hidden)
        )

        return output


class Attention(OnnxOpConverter):
    """Operator converter for Attention from Microsoft onnxruntime contrib opset.

    This is the self-attention mechanism used in transformer models."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        num_heads = attr["num_heads"]

        assert (
            "qkv_hidden_sizes" not in attr
        ), "different hidden sizes for Q, K, V are not currently supported"
        assert "unidirectional" not in attr, "unidirectional attention not current supported"

        # (batch, seq, in_hidden)
        input_emb = inputs[0]

        # (in_hidden, 3 * out_hidden), where out_hidden = num_heads * head_size
        weight = bb.normalize(inputs[1])

        # (3 * out_hidden,)
        bias = bb.normalize(inputs[2])

        # 1. (    batch,              1,        max_seq, max_seq)
        # 2. (    batch, past_seq + seq,)
        # 3. (    batch,            seq, past_seq + seq,)
        # 4. (    batch,)
        # 5. (2 * batch,)
        # For now, we only support case 2.
        mask_index = bb.normalize(inputs[3])

        # (2, batch, num_heads, past_seq, head_size)
        past = inputs[4]

        # (batch, num_heads, seq, seq)
        extra_add = inputs[5]

        (batch_size, seq_len, _) = [val.value for val in input_emb.shape.values]

        (out_hidden_x3,) = [val.value for val in bias.shape.values]
        assert out_hidden_x3 % 3 == 0, "bias shape should be divisible by 3"
        out_hidden = out_hidden_x3 // 3
        assert (
            out_hidden % num_heads == 0
        ), "output hidden size should be divisible by number of attention heads"
        head_size = out_hidden // num_heads

        assert (
            mask_index is not None
        ), "Attention import currently only supports required mask_index"
        mask_index_shape = mask_index.shape
        assert (
            len(mask_index_shape) == 2
            and mask_index_shape[0] == batch_size
            and mask_index_shape[1] == seq_len
        ), "currently only support (batch_size, sequence_length) mask index"

        assert past is None, "past K, V state is not currently supported"
        assert extra_add is None, "extra add to QxK not currently supported"

        split_1 = bb.emit_te(topi.split, weight, 3, 1)
        # split weight and biases and do the matmuls
        w_Q, w_K, w_V = bb.emit(split_1[0]), bb.emit(split_1[1]), bb.emit(split_1[2])

        split_2 = bb.emit_te(topi.split, bias, 3, 0)
        b_Q, b_K, b_V = bb.emit(split_2[0]), bb.emit(split_2[1]), bb.emit(split_2[2])
        # need to merge batch dimensions since TVM matmul is 2D

        # TODO(@yuchen): check reverse_reshape, a hack here
        input_emb = bb.emit_te(
            topi.reshape, input_emb, (input_emb.shape[0] * input_emb.shape[1], input_emb.shape[2])
        )

        mul = bb.emit_te(topi.nn.matmul, input_emb, w_Q)

        Q = bb.emit_te(topi.add, mul, b_Q)

        mul2 = bb.emit_te(topi.nn.matmul, input_emb, w_K)
        K = bb.emit_te(topi.add, mul2, b_K)

        mul3 = bb.emit_te(topi.nn.matmul, input_emb, w_V)
        V = bb.emit_te(topi.add, mul3, b_V)

        # massage tensors in preparation for batched matmul
        def massage(bb, tensor):
            tensor = bb.emit_te(topi.reshape, tensor, (batch_size, seq_len, num_heads, head_size))

            # (batch_size, num_heads, seq_len, head_size)
            tensor = bb.emit_te(topi.transpose, tensor, [0, 2, 1, 3])

            # (batch_size * num_heads, seq_len, head_size)
            # TODO(@yuchen): check reverse_reshape, hack here
            return bb.emit_te(
                topi.reshape,
                tensor,
                (tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3]),
            )

        Q = massage(bb, Q)
        K = massage(bb, K)
        V = massage(bb, V)

        K_present = bb.emit_te(topi.reshape, K, (batch_size, num_heads, seq_len, head_size))
        V_present = bb.emit_te(topi.reshape, V, (batch_size, num_heads, seq_len, head_size))
        present = bb.emit_te(topi.stack, [K_present, V_present], 0)

        att_scores = bb.emit_te(topi.nn.batch_matmul, Q, K, transpose_a=False, transpose_b=True)
        score_dtype = att_scores.checked_type.dtype
        att_scores = bb.emit_te(
            topi.divide,
            att_scores,
            relax.const(np.sqrt(head_size), dtype=att_scores.checked_type.dtype),
        )
        att_scores = bb.emit_te(topi.reshape, att_scores, (batch_size, num_heads, seq_len, seq_len))

        # build the attention mask
        att_mask = bb.emit_te(topi.cast, mask_index, score_dtype)
        att_mask = bb.emit_te(topi.expand_dims, att_mask, 1, num_newaxis=2)
        att_mask = bb.emit_te(topi.subtract, relax.const(1, dtype=score_dtype), att_mask)
        att_mask = bb.emit_te(topi.multiply, att_mask, relax.const(-10000, dtype=score_dtype))

        # apply the mask
        att_scores = bb.emit_te(topi.add, att_scores, att_mask)
        att_scores = bb.emit_te(
            topi.reshape, att_scores, (batch_size * num_heads, seq_len, seq_len)
        )

        att_probs = bb.emit_te(topi.nn.softmax, att_scores, axis=-1)

        output = bb.emit_te(
            topi.nn.batch_matmul, att_probs, V, transpose_a=False, transpose_b=False
        )

        # TODO(@yuchen): check reverse_reshape, hack here
        output = bb.emit_te(
            topi.reshape,
            output,
            (
                int(output.shape[0]) // num_heads,
                num_heads,
                int(output.shape[1]),
                int(output.shape[2]),
            ),
        )

        output = bb.emit_te(topi.transpose, output, axes=[0, 2, 1, 3])
        output = bb.emit_te(
            topi.reshape, output, (int(output.shape[0]), int(output.shape[1]), out_hidden)
        )

        return output


def _get_convert_map(opset):
    return {
        "MatMul": MatMul.get_converter(opset),
        "Tanh": Tanh.get_converter(opset),
        "Relu": Relu.get_converter(opset),
        "Gemm": Gemm.get_converter(opset),
        "Sigmoid": Sigmoid.get_converter(opset),
        "BiasGelu": BiasGelu.get_converter(opset),
        "Gather": Gather.get_converter(opset),
        "Concat": Concat.get_converter(opset),
        "SkipLayerNormalization": SkipLayerNormalization.get_converter(opset),
        "EmbedLayerNormalization": EmbedLayerNormalization.get_converter(opset),
        "Attention": AttentionCutlass.get_converter(opset),
    }


class GraphProto:
    """A helper class for handling Relax expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph
    """

    current = None

    def __init__(self, shape, dtype):
        self._nodes = {}
        self._inputs = {}
        self._num_input = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = dtype
        self.opset = None
        self.bb = relax.BlockBuilder()

    def from_onnx(self, graph, opset) -> IRModule:
        """Construct Relax expression from ONNX graph.

        Onnx graph is a python protobuf object.
        The companion parameters will be handled automatically.
        However, the input names from onnx graph is vague, mixing inputs and
        network weights/bias such as "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        opset : opset version

        Returns
        -------
        mod : tvm.IRModule
            The returned relax module

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        with self.bb.function("main"):
            self.opset = opset
            self._parse_graph_initializers(graph)
            self._parse_graph_input(graph)
            self._check_for_unsupported_ops(graph)
            self._construct_nodes(graph)

            # now return the outputs
            outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
            outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

            ## Maintain the order of inputs and parameters from the ONNX graph, but only include
            ## those parameters that are needed to execute the relax graph
            nodes = {v: k for k, v in self._nodes.items()}
            # Create a function from our output expression and all input variables.
            param_list = [v for k, v in self._inputs.items()]
            self.bb.emit_func_output(outputs, params=param_list)
        return self.bb.get()

    def _parse_graph_initializers(self, graph):
        """Parse network inputs to relax, aka parameters."""
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            self._nodes[init_tensor.name] = relax.const(array)

    def _parse_graph_input(self, graph):
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._nodes:
                continue
            else:
                self._num_input += 1
                self._input_names.append(i_name)
                if i_name in self._shape:
                    i_shape = self._shape[i_name]
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]

    def _check_for_unsupported_ops(self, graph):
        convert_map = _get_convert_map(self.opset)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                # and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def _construct_nodes(self, graph):
        """Nodes are stored as directed acyclic graph."""
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            # Create and populate input list.
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs.append(self._nodes[i])
                else:
                    inputs.append(None)
            i_name = self._parse_value_proto(node)
            outputs = node.output
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(outputs)

            op = self._convert_operator(op_name, inputs, attr, self.opset)
            if not isinstance(op, relax.Tuple):
                outputs_num = 1
            else:
                outputs_num = len(op)

            assert len(outputs) == outputs_num, "Number of output mismatch {} vs {} in {}.".format(
                len(outputs), outputs_num, op_name
            )

            if outputs_num == 1:
                self._nodes[outputs[0]] = op
            else:
                for k, i in zip(list(outputs), range(len(outputs))):
                    self._nodes[k] = op[i]

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_array(self, tensor_proto):
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return tvm.nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ["f", "i", "s", "g"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["t"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["tensors"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["graphs"]:
                if list(getattr(a, f)):
                    raise NotImplementedError("Field {} is not supported in relax.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _convert_operator(self, op_name, inputs, attrs, opset):
        """Convert ONNX operator into a Relax operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relax.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version

        Returns
        -------
        sym : tvm.relax.function.Function
            Converted relax function
        """
        convert_map = _get_convert_map(opset)
        if op_name in convert_map:
            sym = convert_map[op_name](self.bb, inputs, attrs)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_onnx(model, shape=None, dtype="float32", opset=None, convert_config=None):
    """Convert a ONNX model into an equivalent Relax Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    convert_config : Optional[Dict[str, Any]]
        Default config:
            use_nt_batch_matmul : bool = True
                True to convert qualified onnx `matmul` to `nn.batch_matmul` strict to NT format
                (transpose_a=False, transpose_b=True).

    Returns
    -------
    mod : tvm.IRModule
        The relax module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relax
    """
    global ONNX_DEFAULT_CONFIGS
    if convert_config is not None:
        ONNX_DEFAULT_CONFIGS.update(convert_config)

    try:
        import onnx

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except Exception as e:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    g = GraphProto(shape, dtype)
    graph = model.graph

    try:
        opset_in_model = 1
        if model.opset_import:
            # TODO: for now we only really support ai.onnx op set
            # TODO: handle other namespaces well see https://github.com/apache/tvm/issues/10950
            for opset_identifier in model.opset_import:
                # As per https://github.com/onnx/onnx/blob/main/docs/IR.md
                # All operator sets except the default one must specify the operator version
                if str(opset_identifier.domain) in ["ai.onnx", ""]:
                    opset_in_model = opset_identifier.version
                    break
    except AttributeError:
        opset_in_model = 1

    if opset is None:
        opset = opset_in_model
    elif opset < opset_in_model:
        warnings.warn(
            ""
            f"You are overwritting original opset ver = {opset_in_model} by lower ver = {opset}. "
            f"That might cause model conversion errors."
        )

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    return g.from_onnx(graph, opset)

