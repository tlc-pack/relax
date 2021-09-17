/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/expr.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relax {

// call_dps

RELAY_REGISTER_OP("relax.call_dps")
.set_num_inputs(3)
.add_argument("shape", "Expr", "The output shape.")
.add_argument("func", "Expr", "The destination-passing-style function.")
.add_argument("args", "Tuple", "The input arguments.");

Expr MakeCallDPS(Expr shape, Expr func, Tuple args) {
  static const Op& op = Op::Get("relax.call_dps");
  return Call(op, {shape, func, args}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.call_dps").set_body_typed(MakeCallDPS);

// shape_of

RELAY_REGISTER_OP("relax.shape_of")
.set_num_inputs(1)
.add_argument("input", "Expr", "The input expression");

Expr MakeShapeOf(Expr expr) {
  static const Op& op = Op::Get("relax.shape_of");
  return Call(op, {expr}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.shape_of")
.set_body_typed(MakeShapeOf);

// alloc_storage

RELAY_REGISTER_OP("relax.alloc_storage")
.set_num_inputs(1)
.add_argument("size", "Expr", "The size of the storage to allocate.");

Expr MakeAllocStorage(Expr size) {
  static const Op& op = Op::Get("relax.alloc_storage");
  return Call(op, {size}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.alloc_storage")
.set_body_typed(MakeAllocStorage);

// alloc_tensor

RELAY_REGISTER_OP("relax.alloc_tensor")
.set_num_inputs(1)
.add_argument("storage", "Var", "The storage to allocate from.")
.add_argument("offset", "Expr", "The offset into the backing storage.")
.add_argument("shape", "Expr", "The shape of the tensor to allocate.");

Expr MakeAllocTensor(Var storage, Expr offset, Expr shape) {
  static const Op& op = Op::Get("relax.alloc_tensor");
  return Call(op, {storage, offset, shape}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.alloc_tensor")
.set_body_typed(MakeAllocTensor);

// ewise_fma

RELAY_REGISTER_OP("relax.ewise_fma")
.set_num_inputs(3)
.add_argument("e1", "Expr", "The input expression")
.add_argument("e2", "Expr", "The input expression")
.add_argument("e3", "Expr", "The input expression");

Expr MakeEwiseFma(Expr expr1, Expr expr2, Expr expr3) {
  static const Op& op = Op::Get("relax.ewise_fma");
  return Call(op, {expr1, expr2, expr3}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ewise_fma")
.set_body_typed(MakeEwiseFma);
} // namespace relax
} // namespace tvm
