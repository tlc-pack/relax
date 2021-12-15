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
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/op.h>

#include "op_common.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(AllocStorageAttrs);
TVM_REGISTER_NODE_TYPE(AllocTensorAttrs);

bool EqualConstInt(const PrimExpr& lhs, int64_t value) {
  if (const int64_t* pvalue = tir::as_const_int(lhs)) {
    return pvalue[0] == value;
  }
  return false;
}

bool EqualCheck(const PrimExpr& lhs, const PrimExpr& rhs) {
  PrimExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  tvm::arith::Analyzer ana;
  diff = ana.Simplify(diff);
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return false;
}

// call_dps

RELAY_REGISTER_OP("relax.call_dps")
.set_num_inputs(4)
.add_argument("shape", "Expr", "The output shape.")
.add_argument("func", "Expr", "The destination-passing-style function.")
.add_argument("args", "Tuple", "The input arguments.")
.add_argument("packed_ints", "Expr", 
  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from args if unused");

Expr MakeCallDPS(Expr shape, Expr func, Tuple args, Optional<Expr> packed_ints) {
  static const Op& op = Op::Get("relax.call_dps");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {shape, func, args}, {}, {});
  } else {
    call = Call(op, {shape, func, args, packed_ints.value()}, {}, {});
  }
  call->shape_ = shape;
  call->checked_type_ = args->fields[0]->checked_type_;
  return call;
}

TVM_REGISTER_GLOBAL("relax.op.call_dps")
.set_body_typed(MakeCallDPS);

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

// alloc_tensor

RELAY_REGISTER_OP("relax.builtin.alloc_tensor")
.set_num_inputs(1)
.add_argument("shape", "Expr", "The shape of the tensor to allocate.");

Expr MakeAllocTensor(Expr shape) {
  static const Op& op = Op::Get("relax.builtin.alloc_tensor");
  return Call(op, {shape}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.builtin.alloc_tensor")
.set_body_typed(MakeAllocTensor);

// vm alloc_storage

RELAY_REGISTER_OP("relax.vm.builtin.alloc_storage")
.set_attrs_type<AllocStorageAttrs>()
.set_num_inputs(1)
.add_argument("size", "Expr", "The size of the storage to allocate.");

Expr MakeVMAllocStorage(Expr size) {
  static const Op& op = Op::Get("relax.vm.builtin.alloc_storage");
  return Call(op, {size}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.alloc_storage")
.set_body_typed(MakeVMAllocStorage);

// vm alloc_tensor

RELAY_REGISTER_OP("relax.vm.builtin.alloc_tensor")
.set_attrs_type<AllocTensorAttrs>()
.set_num_inputs(1)
.add_argument("shape", "Expr", "The shape of the tensor to allocate.");

Expr MakeVMAllocTensor(Expr shape) {
  static const Op& op = Op::Get("relax.vm.builtin.alloc_tensor");
  return Call(op, {shape}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.alloc_tensor")
.set_body_typed(MakeVMAllocTensor);

// vm store_shape

RELAY_REGISTER_OP("relax.vm.builtin.store_shape")
.set_attrs_type<ShapeHeapAttrs>()
.set_num_inputs(2)
.add_argument("shape", "Expr", "The shape to be stored.")
.add_argument("heap", "Expr", "The heap to store the shape.");

Expr MakeStoreShape(Expr shape, Expr heap) {
  static const Op& op = Op::Get("relax.vm.builtin.store_shape");
  return Call(op, {shape, heap}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.store_shape")
.set_body_typed(MakeStoreShape);

// vm load_shape

RELAY_REGISTER_OP("relax.vm.builtin.load_shape")
.set_attrs_type<ShapeHeapAttrs>()
.set_num_inputs(1)
.add_argument("heap", "Expr", "The heap to load the shape from.");

Expr MakeLoadShape(Expr heap) {
  static const Op& op = Op::Get("relax.vm.builtin.load_shape");
  return Call(op, {heap}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.load_shape")
.set_body_typed(MakeLoadShape);

// vm call_tir_dyn
RELAY_REGISTER_OP("relax.vm.call_tir_dyn")
.set_num_inputs(2)
.add_argument("func", "Expr", "The destination-passing-style function.")
.add_argument("args", "Tuple", "The input arguments (list of tensors and last argument is ShapeExpr)");


} // namespace relax
} // namespace tvm
