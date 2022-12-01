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
#include <tvm/relax/utils.h>
#include <tvm/relay/op.h>

#include "op_common.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(AllocTensorAttrs);
TVM_REGISTER_NODE_TYPE(MemAllocStorageAttrs);
TVM_REGISTER_NODE_TYPE(MemAllocTensorAttrs);
TVM_REGISTER_NODE_TYPE(VMAllocStorageAttrs);
TVM_REGISTER_NODE_TYPE(VMAllocTensorAttrs);
TVM_REGISTER_NODE_TYPE(ShapeHeapAttrs);

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

Type ReturnVoidType(const Call& call, DiagnosticContext diag_ctx) { return VoidType(); }

Type ReturnObjectType(const Call& call, DiagnosticContext diag_ctx) { return ObjectType(); }

Type ReturnShapeType(const Call& call, DiagnosticContext diag_ctx) { return ShapeType(); }

// call_tir

Expr InferShapeCallTIR(const Call& call, DiagnosticContext diag_ctx) {
  Expr output_shape = call->args[2];
  return output_shape;
}

Type InferTypeArg(const Call& call, DiagnosticContext diag_ctx) {
  if (call->type_args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "type_args should have exact 1 output type.");
  }
  Type output_type = call->type_args[0];
  return output_type;
}

RELAY_REGISTER_OP("relax.call_tir")
    .set_num_inputs(4)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("output_shape", "Expr", "The output shape.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferShape>("FInferShape", InferShapeCallTIR)
    .set_attr<FInferType>("FInferType", InferTypeArg);

Expr MakeCallTIR(Expr func, Tuple args, Expr output_shape, Type output_type,
                 Optional<Expr> packed_ints) {
  static const Op& op = Op::Get("relax.call_tir");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args, output_shape}, {}, {output_type});
  } else {
    call = Call(op, {func, args, output_shape, packed_ints.value()}, {}, {output_type});
  }
  return call;
}

TVM_REGISTER_GLOBAL("relax.op.call_tir").set_body_typed(MakeCallTIR);

// print
TVM_REGISTER_NODE_TYPE(PrintAttrs);

RELAY_REGISTER_OP("relax.print")
    .set_attrs_type<PrintAttrs>()
    .set_num_inputs(-1)
    .add_argument("vals", "Array<Expr>", "Values to print.")
    .set_attr<FInferType>("FInferType", ReturnVoidType)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.print");

Expr MakePrint(Array<Expr> vals, std::string format) {
  auto attrs = make_object<PrintAttrs>();
  attrs->format = format;
  static const Op& op = Op::Get("relax.print");
  return Call(op, vals, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.print").set_body_typed(MakePrint);

// assert_op

// can't actually name it assert or else Python will consider it a syntax error

Type InferAssertType(const Call& call, DiagnosticContext diag_ctx) {
  // Ensure that the condition argument is a boolean scalar.
  // Also permitted is a tensor with unknown shape and unknown dtype
  // (checked dynamically in that case). Returns void.
  if (call->args.size() < 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Assert must have at least one argument (the condition).");
  }
  Type arg_type = call->args[0]->checked_type();
  if (!IsBoolScalarType(arg_type)) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The argument to assert must be a boolean scalar type, but received "
                       << arg_type);
  }
  return VoidType();
}

TVM_REGISTER_NODE_TYPE(AssertOpAttrs);

RELAY_REGISTER_OP("relax.assert_op")
    .set_attrs_type<AssertOpAttrs>()
    .set_num_inputs(-1)
    .add_argument("vals", "Array<Expr>",
                  "The first value is used as the assertion condition. The others are used as "
                  "format arguments if there is an error.")
    .set_attr<FInferType>("FInferType", InferAssertType)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.assert_op");

Expr MakeAssertOp(Expr condition, Array<Expr> vals, std::string format) {
  auto attrs = make_object<AssertOpAttrs>();
  attrs->format = format;
  static const Op& op = Op::Get("relax.assert_op");
  Array<Expr> args = {condition};
  for (auto val : vals) {
    args.push_back(val);
  }
  return Call(op, args, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.assert_op").set_body_typed(MakeAssertOp);

// make_closure

RELAY_REGISTER_OP("relax.make_closure")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The closure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferType>("FInferType", ReturnObjectType);

Expr MakeClosure(Expr func, Tuple args) {
  static const Op& op = Op::Get("relax.make_closure");
  return Call(op, {func, args}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.make_closure").set_body_typed(MakeClosure);

// invoke_closure

RELAY_REGISTER_OP("relax.invoke_closure")
    .set_num_inputs(2)
    .add_argument("closure", "Expr", "The VMClosure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferType>("FInferType", InferTypeArg);

Expr InvokeClosure(Expr closure, Tuple args, Array<Type> type_args) {
  static const Op& op = Op::Get("relax.invoke_closure");
  return Call(op, {closure, args}, {}, type_args);
}

TVM_REGISTER_GLOBAL("relax.op.invoke_closure").set_body_typed(InvokeClosure);

// shape_of

RELAY_REGISTER_OP("relax.shape_of")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferType>("FInferType", ReturnShapeType);

Expr MakeShapeOf(Expr expr) {
  static const Op& op = Op::Get("relax.shape_of");
  return Call(op, {expr}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.shape_of").set_body_typed(MakeShapeOf);

// alloc_tensor

Expr InferShapeAllocTensor(const Call& call, DiagnosticContext diag_ctx) { return call->args[0]; }

Type InferTypeAllocTensor(const Call& call, DiagnosticContext diag_ctx) {
  auto attrs = call->attrs.as<AllocTensorAttrs>();
  ICHECK(attrs != nullptr) << "must be AllocTensorAttrs, but got " << call->attrs->GetTypeKey();
  auto output_shape = call->args[0].as<ShapeExprNode>();
  ICHECK(output_shape != nullptr) << "must be ShapeExpr, but got " << call->args[0]->GetTypeKey();
  return DynTensorType(output_shape->values.size(), attrs->dtype);
}

RELAY_REGISTER_OP("relax.builtin.alloc_tensor")
    .set_attrs_type<AllocTensorAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .set_attr<FInferShape>("FInferShape", InferShapeAllocTensor)
    .set_attr<FInferType>("FInferType", InferTypeAllocTensor);

Expr MakeAllocTensor(Expr shape, DataType dtype, int64_t runtime_device_index) {
  auto attrs = make_object<AllocTensorAttrs>();
  attrs->dtype = std::move(dtype);
  attrs->runtime_device_index = std::move(runtime_device_index);
  static const Op& op = Op::Get("relax.builtin.alloc_tensor");
  return Call(op, {shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.builtin.alloc_tensor").set_body_typed(MakeAllocTensor);

// memory planning alloc_storage

RELAY_REGISTER_OP("relax.memory.alloc_storage")
    .set_attrs_type<MemAllocStorageAttrs>()
    .set_num_inputs(1)
    .add_argument("total_space", "Expr", "The total space of the storage to allocate.")
    .set_attr<FInferType>("FInferType", ReturnObjectType);

Expr MakeAllocStorage(Expr size, int64_t virtual_device_index, std::string storage_scope,
                      DataType dtype) {
  auto attrs = make_object<MemAllocStorageAttrs>();
  attrs->virtual_device_index = std::move(virtual_device_index);
  attrs->storage_scope = std::move(storage_scope);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("relax.memory.alloc_storage");
  return Call(op, {size}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.alloc_storage").set_body_typed(MakeAllocStorage);

// memory planning alloc_tensor

Expr InferShapeMemAllocTensor(const Call& call, DiagnosticContext diag_ctx) {
  return call->args[1];
}

Type InferTypeMemAllocTensor(const Call& call, DiagnosticContext diag_ctx) {
  auto attrs = call->attrs.as<MemAllocTensorAttrs>();
  ICHECK(attrs != nullptr) << "must be MemAllocTensorAttrs , but got " << call->attrs->GetTypeKey();
  if (const auto* output_shape = call->args[1].as<ShapeExprNode>()) {
    return DynTensorType(output_shape->values.size(), attrs->dtype);
  }
  return DynTensorType::CreateUnknownNDim(attrs->dtype, Span());
}

RELAY_REGISTER_OP("relax.memory.alloc_tensor")
    .set_attrs_type<MemAllocTensorAttrs>()
    .set_num_inputs(2)
    .add_argument("storage", "Expr", "The storage to allocate the tensor to.")
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .set_attr<FInferShape>("FInferShape", InferShapeMemAllocTensor)
    .set_attr<FInferType>("FInferType", InferTypeMemAllocTensor);

Expr MakeMemAllocTensor(Expr storage, Expr shape, int offset, DataType dtype) {
  auto attrs = make_object<MemAllocTensorAttrs>();
  attrs->offset = std::move(offset);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("relax.memory.alloc_tensor");
  return Call(op, {storage, shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.alloc_tensor").set_body_typed(MakeMemAllocTensor);

// memory planning kill_storage

RELAY_REGISTER_OP("relax.memory.kill_storage")
    .set_num_inputs(1)
    .add_argument("storage", "Expr", "The storage to be killed.")
    .set_attr<FInferType>("FInferType", ReturnVoidType);

Expr MakeMemKillStorage(Expr storage) {
  static const Op& op = Op::Get("relax.memory.kill_storage");
  return Call(op, {storage}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.kill_storage").set_body_typed(MakeMemKillStorage);

// memory planning kill_tensor

RELAY_REGISTER_OP("relax.memory.kill_tensor")
    .set_num_inputs(1)
    .add_argument("tensor", "Expr", "The tensor to be killed.")
    .set_attr<FInferType>("FInferType", ReturnVoidType);

Expr MakeMemKillTensor(Expr tensor) {
  static const Op& op = Op::Get("relax.memory.kill_tensor");
  return Call(op, {tensor}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.kill_tensor").set_body_typed(MakeMemKillTensor);

// vm alloc_storage

RELAY_REGISTER_OP("relax.vm.builtin.alloc_storage")
    .set_attrs_type<VMAllocStorageAttrs>()
    .set_num_inputs(1)
    .add_argument("size", "Expr", "The size of the storage to allocate.")
    .set_attr<FInferType>("FInferType", ReturnObjectType);

Expr MakeVMAllocStorage(Expr size, DataType dtype, int64_t runtime_device_index) {
  auto attrs = make_object<VMAllocStorageAttrs>();
  attrs->dtype = std::move(dtype);
  attrs->runtime_device_index = std::move(runtime_device_index);
  static const Op& op = Op::Get("relax.vm.builtin.alloc_storage");
  return Call(op, {size}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.alloc_storage").set_body_typed(MakeVMAllocStorage);

// vm alloc_tensor

Expr InferShapeVMAllocTensor(const Call& call, DiagnosticContext diag_ctx) { return call->args[1]; }

Type InferTypeVMAllocTensor(const Call& call, DiagnosticContext diag_ctx) {
  auto attrs = call->attrs.as<VMAllocTensorAttrs>();
  ICHECK(attrs != nullptr) << "must be VMAllocTensorAttrs , but got " << call->attrs->GetTypeKey();
  if (const auto* output_shape = call->args[1].as<ShapeExprNode>()) {
    return DynTensorType(output_shape->values.size(), attrs->dtype);
  }
  return DynTensorType::CreateUnknownNDim(attrs->dtype, Span());
}

RELAY_REGISTER_OP("relax.vm.builtin.alloc_tensor")
    .set_attrs_type<VMAllocTensorAttrs>()
    .set_num_inputs(2)
    .add_argument("storage", "Expr", "The storage to allocate the tensor to.")
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .set_attr<FInferShape>("FInferShape", InferShapeVMAllocTensor)
    .set_attr<FInferType>("FInferType", InferTypeVMAllocTensor);

Expr MakeVMAllocTensor(Expr storage, Expr shape, int offset, DataType dtype) {
  auto attrs = make_object<VMAllocTensorAttrs>();
  attrs->offset = std::move(offset);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("relax.vm.builtin.alloc_tensor");
  return Call(op, {storage, shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.alloc_tensor").set_body_typed(MakeVMAllocTensor);

// vm store_shape

RELAY_REGISTER_OP("relax.vm.builtin.store_shape")
    .set_attrs_type<ShapeHeapAttrs>()
    .set_num_inputs(2)
    .add_argument("shape", "Expr", "The shape to be stored.")
    .add_argument("heap", "Expr", "The heap to store the shape.")
    .set_attr<FInferType>("FInferType", ReturnVoidType);

Expr MakeStoreShape(Expr shape, Expr heap, Array<Integer> indices) {
  auto attrs = make_object<ShapeHeapAttrs>();
  attrs->indices = std::move(indices);
  static const Op& op = Op::Get("relax.vm.builtin.store_shape");
  return Call(op, {shape, heap}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.store_shape").set_body_typed(MakeStoreShape);

// vm load_shape

RELAY_REGISTER_OP("relax.vm.builtin.load_shape")
    .set_attrs_type<ShapeHeapAttrs>()
    .set_num_inputs(1)
    .add_argument("heap", "Expr", "The heap to load the shape from.")
    .set_attr<FInferType>("FInferType", ReturnShapeType);

Expr MakeLoadShape(Expr heap, Array<Integer> indices) {
  auto attrs = make_object<ShapeHeapAttrs>();
  attrs->indices = std::move(indices);
  static const Op& op = Op::Get("relax.vm.builtin.load_shape");
  return Call(op, {heap}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.builtin.load_shape").set_body_typed(MakeLoadShape);

// vm call_tir_dyn

RELAY_REGISTER_OP("relax.vm.call_tir_dyn")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple",
                  "The input arguments (list of tensors and last argument is ShapeExpr)")
    .set_attr<FInferType>("FInferType", ReturnVoidType);

}  // namespace relax
}  // namespace tvm
