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
"""Developer API of building Relax IR nodes."""

from tvm import relax as rx
from tvm.relay.expr import Tuple, Call
from .expr import *


class FunctionScope(object):
    """Auxiliary scope for function"""

    stack = []
    functions = []

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.global_bindings = []
        self.binding_blocks = []
        self.body = None

    def __enter__(self):
        FunctionScope.stack.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        if len(self.global_bindings) > 0:
            binding_block = BindingBlock(self.global_bindings)
            self.binding_blocks.append(binding_block)
            self.global_bindings.clear()
        seq = SeqExpr(self.binding_blocks, self.body)
        FunctionScope.functions.append(Function(self.params, seq, None, rx.GlobalVar(self.name)))


class DataflowScope(object):
    """Auxiliary scope for Dataflow block"""

    stack = []

    def __init__(self):
        self.bindings = []

    def __enter__(self):
        DataflowScope.stack.append(self)
        if len(FunctionScope.stack[-1].global_bindings) > 0:
            binding_block = BindingBlock(FunctionScope.stack[-1].global_bindings)
            FunctionScope.stack[-1].binding_blocks.append(binding_block)
            FunctionScope.stack[-1].global_bindings.clear()
        return self

    def __exit__(self, ptype, value, trace):
        DataflowScope.stack.pop()
        FunctionScope.stack[-1].binding_blocks.append(DataflowBlock(self.bindings))


class IRBuilder(object):
    """Builder to construct Relax AST for testing and dev.
    Examples
    --------
    .. code-block:: python

        ib = rx.ir_builder.create()
        x = ib.var("x", shape, dtype)
        with ib.function("foo", [x]):
            with ib.dataflow():
                y = ib.dataflow_var("y", shape, dtype)
                z = ib.dataflow_var("z", shape, dtype)
                ib.bind(z, ib.call(add_func, [x, y]))
                res = ib.var("res")
                ib.bind(res, ib.call(mul_func, [x, z]))
            ib.output(res)
        ib.get()
    """

    def __init__(self):
        self.var_bindings = []

    @staticmethod
    def _check_dataflow_scope():
        if len(DataflowScope.stack) == 0:
            raise ValueError("Dataflow variable creation should happen in a dataflow scope")

    @staticmethod
    def _check_function_scope():
        if len(FunctionScope.stack) == 0:
            raise ValueError("This method should happen in a function scope")

    def var(self, name, shape=None, dtype=None):
        """Create a global Var.

        Parameters
        ----------
        name : str
            The name of the variable

        shape : Optional[List[Type]]
            The shape annotation of the variable

        dtype : Optional[Type]
            The data type of the variable

        Returns
        -------
        var : tvm.relax.Var
        """
        return Var(name, shape, dtype)

    def dataflow_var(self, name, shape=None, dtype=None):
        """Create a DataflowVar.

        Parameters
        ----------
        name : str
            The name of the variable

        shape : Optional[List[Type]]
            The shape annotation of the variable

        dtype : Optional[Type]
            The data type of the variable

        Returns
        -------
        var : tvm.relax.DataflowVar
        """
        self._check_dataflow_scope()
        return DataflowVar(name, shape, dtype)

    def bind(self, var, value):
        """Create a binding.

        Parameters
        ----------
        var : Var
            The variable

        value : Expr
            The value to be bound
        """
        self._check_function_scope()
        if len(DataflowScope.stack) != 0:
            DataflowScope.stack[-1].bindings.append(VarBinding(var, value))
        else:
            FunctionScope.stack[-1].global_bindings.append(VarBinding(var, value))

    def call(self, op, args=None, attrs=None, type_args=None):
        """Create a function call node。

        Parameters
        ----------
        op: tvm.ir.Op or any Expr with function type.
            The operation to be called.

        args: List[tvm.relay.Expr]
            The arguments to the call.

        attrs: Optional[tvm.Attrs]
            Attributes to the call, can be None

        type_args: Optional[List[Type]]
            The additional type arguments, this is only
            used in advanced usecase of template functions.
        """
        return Call(op, args, attrs, type_args)

    def dataflow(self):
        """Create a DataflowBlock."""
        self._check_function_scope()
        return DataflowScope()

    def function(self, name, params):
        """Create a Function.

        Parameters
        ----------
        name : str
            The name of the function

        params :
            The parameters of the function
        """
        if not isinstance(params, (list, tuple)):
            params = [params]
        return FunctionScope(name, params)

    def output(self, output):
        """Specify the outputs of a function.

        Parameters
        ----------
        output : str
            The name of the function
        """
        self._check_function_scope()
        if isinstance(output, Expr):
            FunctionScope.stack[-1].body = output
        elif isinstance(output, (list, tuple)):
            FunctionScope.stack[-1].body = Tuple(output)
        else:
            raise ValueError("output must be an Expr or a list/tuple of Expr")

    def get(self):
        """Return the built AST."""
        return FunctionScope.functions[-1]


def create():
    """Create a new IRBuilder

    Returns
    -------
    builder : IRBuilder
        The created IRBuilder
    """
    return IRBuilder()
