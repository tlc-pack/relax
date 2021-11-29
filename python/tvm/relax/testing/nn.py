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

import tvm
from typing import List, Optional, Union, Dict, Any, Callable
from tvm.relay import Call
from tvm import relax, topi


class FunctionScope(object):
    """Auxiliary scope for relax function"""

    def __init__(self):
        self._ib = current_builder()

    def __enter__(self):
        self._ib._begin_binding_block()

    def __exit__(self, ptype, value, trace):
        self._ib._blocks = []


class Builder:
    """A builder to build Relax IR with a pytorch-like nn.Module API."""

    current = None

    def __init__(self):
        Builder.current = relax.BlockBuilder()

    def func(self, name: str) -> FunctionScope:
        Builder.current._func_name = name
        return FunctionScope()

    def finalize(self, args, result):
        if Builder.current:
            block = Builder.current._end_block()
            if len(block.bindings) > 0:
                Builder.current._blocks.append(block)
            seqe = relax.SeqExpr(Builder.current._blocks, result)
            gvar = relax.GlobalVar(Builder.current._func_name)
            func = relax.Function(args, seqe, relax.DynTensorType(-1, "float32"), gvar)
            gvar = relax.GlobalVar(Builder.current._func_name)
            Builder.current._context_mod[gvar] = func
            return func
        else:
            raise ValueError("block builder has not been initialized")

    def get(self):
        return Builder.current.get()


def current_builder():
    return Builder.current


def emit_te(func: Callable, *args: Any, **kwargs: Any) -> relax.Var:
    return current_builder().emit_te(func, *args, **kwargs)


class Placeholder(relax.Var):
    """A placeholder variable that can represent model input."""

    def __init__(self, shape, dtype="float32", name="data"):
        if isinstance(shape, (list, tuple)):
            rank = len(shape)
        type_anno = relax.DynTensorType(rank, dtype)
        super().__init__(current_builder()._get_unique_name(name), shape, type_anno)


class Parameter(relax.Var):
    """A special kind of relax Var that represents model parameter(weight)."""

    def __init__(self, shape, dtype="float32", name="param"):
        if isinstance(shape, (list, tuple)):
            rank = len(shape)
        # TODO: handle other cases
        type_anno = relax.DynTensorType(rank, dtype)
        super().__init__(current_builder()._get_unique_name(name), shape, type_anno)


class Module:
    """Base class for all model modules.

    A neural network or a layer can subclass this class.
    """

    def parameters(self) -> List[Parameter]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _unpack_params(value: object) -> List[relax.Var]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


class Sequential(Module):
    """A sequential container that concatenates modules in it.

    Example
    -------

    .. code-block:: python

    model = nn.Sequential(
                nn.Conv2d(1, 20, 5),
                nn.ReLU(),
                nn.Conv2d(20, 64, 5),
                nn.ReLU()
            )
    """

    def __init__(self, *modules: Module):
        self.modules = modules

    def forward(self, input: relax.Var) -> relax.Var:
        for module in self.modules:
            input = module(input)
        return input


class ReLU(Module):
    """Applies the rectified linear unit activation function on the input."""

    def forward(self, input: relax.Var) -> relax.Var:
        return emit_te(topi.nn.relu, input)


class LogSoftmax(Module):
    """Applies log softmax activation function on the input."""

    def forward(self, input: relax.Var) -> relax.Var:
        return emit_te(topi.nn.log_softmax, input)


class Linear(Module):
    """Applies a linear transformation to the input data: :math:`y = xA + b`."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((in_features, out_features), name="linear_weight")
        if bias:
            self.bias = Parameter((out_features,), name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Var) -> relax.Var:
        y = emit_te(topi.matmul, input, self.weight)
        if self.bias is not None:
            y = emit_te(topi.add, y, self.bias)
        return y
