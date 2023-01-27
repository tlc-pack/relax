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
from typing import List, Dict, Callable, Tuple
from numpy import iterable
import os

import torch
from torch import nn, fx

import tvm
from tvm import relax, topi
import numpy as np
import operator


class TorchFXImporter:
    def __init__(self, module: fx.GraphModule) -> None:
        self.env = {}
        self.params = {}
        self.params_transpose = {}
        self.named_modules = dict(module.named_modules())
        self.bb = relax.BlockBuilder()
        self.create_convert_map()
        self.missing_info = dict()

    @staticmethod
    def _convert_data_type(input_type):
        """converts the PyTorch scalar type input_type to a TVM dtype."""

        input_type = input_type.lower()
        if input_type in ["double", "float64", "torch.float64"]:
            return "float64"
        elif input_type in ["float", "float32", "torch.float32"]:
            return "float32"
        elif input_type in ["half", "float16", "torch.float16"]:
            return "float16"
        elif input_type in ["long", "int64", "torch.int64"]:
            return "int64"
        elif input_type in ["int", "int32", "torch.int32"]:
            return "int32"
        elif input_type in ["short", "int16", "torch.int16"]:
            return "int16"
        elif input_type in ["char", "int8", "torch.int8"]:
            return "int8"
        elif input_type in ["byte", "uint8", "torch.uint8"]:
            return "uint8"
        elif input_type in ["quint8", "torch.quint8"]:
            return "quint8"
        elif input_type in ["qint8", "torch.qint8"]:
            return "qint8"
        elif input_type in ["qint32", "torch.qint32"]:
            return "qint32"
        elif input_type in ["bool", "torch.bool"]:
            return "bool"
        elif input_type in ["str"]:
            return "str"
        else:
            raise NotImplementedError("input_type {} is not handled yet".format(input_type))

    @staticmethod
    def _fetch_attr(model, target: str):
        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        if isinstance(attr_itr, torch.Tensor):
            return TorchFXImporter._convert_torch_tensor_to_relax(attr_itr)
        return attr_itr

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        ndim = len(tensor.data.shape)
        dtype = TorchFXImporter._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.cpu().numpy(), relax.DynTensorType(ndim, dtype))

    def retrive_args(self, node):
        return self._retrive_args(node.args)

    def _retrive_args(self, node):
        if isinstance(node, fx.node.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrive_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrive_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrive_args(k): self._retrive_args(v) for k, v in node.items()}
        else:
            return node

    @staticmethod
    def _promote_binary_op_args(lhs, rhs):
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return lhs, rhs
        elif isinstance(lhs, relax.Expr):
            assert isinstance(lhs.checked_type, relax.DynTensorType)
            return lhs, relax.const(rhs, lhs.checked_type.dtype)
        elif isinstance(rhs, relax.Expr):
            assert isinstance(rhs.checked_type, relax.DynTensorType)
            return relax.const(lhs, rhs.checked_type.dtype), rhs
        else:
            assert False

    def _call_binary_op(self, op, lhs, rhs):
        lhs, rhs = TorchFXImporter._promote_binary_op_args(lhs, rhs)
        return self.bb.emit(op(lhs, rhs))

    def _add(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.add, lhs, rhs)
        return lhs + rhs

    def _mul(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.multiply, lhs, rhs)
        return lhs * rhs

    def _getitem(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]

        if iterable(x):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.shape, relax.Tuple):
                return self.bb.emit(relax.TupleGetItem(x, node.args[1]))
            else:
                begin = []
            end = []
            stride = []
            axes = []
            expand_dim = []
            i = 0
            for index in node.args[1]:
                if isinstance(index, int):
                    begin.append(index)
                    end.append(index + 1)
                    stride.append(1)
                    axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    begin.append(0 if index.start is None else index.start)
                    end.append(x.shape_[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(i)
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(x.shape_):
                begin.append(0)
                end.append(x.shape_[i])
                axes.append(i)
                i = i + 1
            sliced = self.bb.emit_te(topi.strided_slice, x, begin, end, stride, axes)
            sliced_shape = list(sliced.shape_)
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.bb.emit(relax.op.reshape(sliced, sliced_shape))
        else:
            raise Exception(f"Pleaes check the tensor: {x}")

    # TODO(@tvm-team): Currently, only supports a few operators for fallback mechanism demo
    def create_convert_map(self):
        self.convert_map = {
            # Torch operators
            torch.add: self._add,
            torch.mul: self._mul,
            # Python builtin operators
            operator.add: self._add,
            operator.mul: self._mul,
            operator.getitem: self._getitem,
        }


def from_pytorch(model: torch.nn.Module, input_infos: Dict[str, Tuple]):
    symbolic_traced: fx.GraphModule = fx.symbolic_trace(model)
    importer = TorchFXImporter(symbolic_traced)
    graph = symbolic_traced.graph

    # Extract input names from the graph
    graph_input_names = [node.name for node in graph.nodes if node.op == "placeholder"]

    inputs = {}
    for graph_input_name, (user_assigned_name, (shape, dtype)) in zip(
        graph_input_names, input_infos.items()
    ):
        inputs[graph_input_name] = relax.Var(
            user_assigned_name, relax.TensorStructInfo(shape, dtype)
        )

    # Translate model parameters.
    for _, param in model.named_parameters():
        ndim = len(param.data.shape)
        dtype = importer._convert_data_type(str(param.data.dtype))
        importer.params[param] = relax.const(
            param.data.cpu().numpy(), relax.TensorStructInfo(ndim=ndim, dtype=dtype)
        )

    # Initialize the block builder with a function and a dataflow block.
    # Construct the relax "main" function that calls each of submodule functions we created.
    bb = importer.bb
    ext_mods = list()
    with bb.function(name="main", params=list(inputs.values())):
        output = None
        with bb.dataflow():
            for node in graph.nodes:
                if node.op == "placeholder":
                    assert node.name in inputs, "The function input {} is not found".format(
                        node.name
                    )
                    importer.env[node] = inputs[node.name]
                elif node.op == "output":
                    output = bb.emit_output(importer.env[node.args[0]])
                    break
                elif node.op == "get_attr":
                    importer.env[node] = TorchFXImporter._fetch_attr(model, node.target)
                elif node.op == "call_module":
                    module = importer.named_modules[node.target]
                    assert (
                        node.target in importer.convert_map
                    ), f"Unsupported module type {type(module)}"
                    importer.env[node] = importer.convert_map[node.target](node)
                elif node.op == "call_function":
                    func_name = node.name.rstrip("0123456789_")
                    assert (
                        node.target in importer.convert_map
                    ), f"Unsupported function type {func_name}"
                    importer.env[node] = importer.convert_map[node.target](node)
                elif node.op == "call_method":
                    assert (
                        node.target in importer.convert_map
                    ), f"Unsupported function target {node.target}"
                    importer.env[node] = importer.convert_map[node.target](node)
                else:
                    raise ValueError(f"Unsupported op {node.op}")

        assert output is not None
        bb.emit_func_output(output)
    return bb.get()
