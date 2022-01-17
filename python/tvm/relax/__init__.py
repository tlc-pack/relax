# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from . import exec_builder
from . import expr
from . import ty
from . import vm
from . import block_builder
from . import op
from . import analysis
from . import transform


# Expr
Expr = expr.Expr
Span = expr.Span
SourceName = expr.SourceName
Id = expr.Id
GlobalVar = expr.GlobalVar
Var = expr.Var
DataflowVar = expr.DataflowVar
Binding = expr.Binding
MatchShape = expr.MatchShape
VarBinding = expr.VarBinding
BindingBlock = expr.BindingBlock
DataflowBlock = expr.DataflowBlock
SeqExpr = expr.SeqExpr
ShapeExpr = expr.ShapeExpr
Tuple = expr.Tuple
TupleGetItem = expr.TupleGetItem
Function = expr.Function
ExternFunc = expr.ExternFunc
Call = expr.Call
If = expr.If

# helper functions
const = expr.const
extern = expr.extern
te_tensor = expr.te_tensor

# Type
Type = ty.Type
ShapeType = ty.ShapeType
DynTensorType = ty.DynTensorType
DimType = ty.DimType

# VM
ExecBuilder = exec_builder.ExecBuilder
VirtualMachine = vm.VirtualMachine
load_exec_from_file = vm.load_exec_from_file

# Operator
from .op.base import call_tir
from .op.op_attrs import AllocStorageAttrs, AllocTensorAttrs

# IRBuilder
BlockBuilder = block_builder.BlockBuilder
