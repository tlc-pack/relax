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

#include <tvm/arith/analyzer.h>
#include <tvm/relax/type.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

using FInferShape = runtime::TypedPackedFunc<Optional<RelayExpr>(relay::Call call)>;

using FInferType = runtime::TypedPackedFunc<Type(Array<Type>)>;

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

void BinaryBroadcast(TVMArgs args, TVMRetValue* rv) {
  ShapeTuple lhs_shape = args[0];
  ShapeTuple rhs_shape = args[0];
  std::vector<int64_t> output_shape;
  size_t ndim0 = lhs_shape.size();
  size_t ndim1 = rhs_shape.size();
  size_t i = 1;
  for (; i <= std::min(ndim0, ndim1); ++i) {
    int64_t lhs_dim = lhs_shape[ndim0 - i];
    int64_t rhs_dim = rhs_shape[ndim1 - i];
    if (lhs_dim == 1 || rhs_dim == 1 || lhs_dim == rhs_dim) {
      output_shape.push_back(std::max(lhs_dim, rhs_dim));
    } else {
      LOG(FATAL) << "Incompatible shapes " << lhs_shape << " and " << rhs_shape
                 << " for broadcasting";
    }
    size_t max_ndim = std::max(ndim0, ndim1);
    ShapeTuple& longer_shape = (ndim0 > ndim1) ? lhs_shape : rhs_shape;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape[max_ndim - i]);
    }
    *rv = ShapeTuple(output_shape);
  }
}

Optional<RelayExpr> InferShapeBinaryBroadcast(relay::Call call) {
  ICHECK_EQ(call->args.size(), 2);
  auto lhs_shape = call->args[0]->shape_;
  auto rhs_shape = call->args[1]->shape_;
  if (lhs_shape && rhs_shape) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = lhs_shape.value().size();
    size_t ndim1 = rhs_shape.value().size();
    size_t i = 1;
    for (; i <= std::min(ndim0, ndim1); ++i) {
      PrimExpr s1 = lhs_shape.value()[ndim0 - i];
      PrimExpr s2 = rhs_shape.value()[ndim1 - i];
      if (EqualConstInt(s1, 1)) {
        output_shape.push_back(s2);
      } else if (EqualConstInt(s2, 1)) {
        output_shape.push_back(s1);
      } else if (EqualCheck(s1, s2)) {
        output_shape.push_back(s1);
      } else {
        // defer the computation of output shapes to runtime
        // e.g., [m, n]  [l, k] -> defer to runtime
        return PackedFunc(BinaryBroadcast)(lhs_shape, rhs_shape);

        // [m+5]  [m] definitely cannot be match the rule -> FATAL
        // LOG(FATAL) << "Incompatible shapes " << lhs_shape << " and " << rhs_shape
        //            << " for broadcasting";
      }
    }
    size_t max_ndim = std::max(ndim0, ndim1);
    auto& longer_shape = (ndim0 > ndim1) ? lhs_shape : rhs_shape;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape.value()[max_ndim - i]);
    }
    return ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
  } else {
    return NullOpt;
  }
}

Type InferTypeBinaryBroadcast(Array<Type> args) {
  Type lhs_type = args[0];
  Type rhs_type = args[1];
  if (auto* t0 = lhs_type.as<DynTensorTypeNode>()) {
    if (auto* t1 = rhs_type.as<DynTensorTypeNode>()) {
      DataType outout_dtype;
      if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
        outout_dtype = DataType::Void();
      } else if (t0->dtype != t1->dtype) {
        LOG(FATAL) << "Data types " << t0->dtype << " and " << t1->dtype
                   << " do not match broadcasting rule";
      } else {
        outout_dtype = t0->dtype;
      }

      int output_rank;
      if (t0->IsUnknownRank() || t1->IsUnknownRank()) {
        output_rank = -1;
      } else {
        output_rank = std::max(t0->rank, t1->rank);
      }
      return DynTensorType(output_rank, outout_dtype);
    }
  }
  LOG(FATAL) << "Both lhs and rhs should be DynTensor for broadcasting";
  return VoidType();
}

Expr Add(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("relax.add");
  return relay::Call(op, {lhs, rhs}, {}, {});
}

Expr Multiply(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("relax.multiply");
  return relay::Call(op, {lhs, rhs}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op._make.add").set_body_typed(Add);
TVM_REGISTER_GLOBAL("relax.op._make.multiply").set_body_typed(Multiply);

TVM_REGISTER_OP("relax.add")
    .describe("Elementwise add with broadcasting")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side tensor.")
    .add_argument("rhs", "Tensor", "The right hand side tensor.")
    .set_support_level(1)
    .set_attr<FInferShape>("FInferShape", InferShapeBinaryBroadcast)
    .set_attr<FInferType>("FInferType", InferTypeBinaryBroadcast);

TVM_REGISTER_OP("relax.multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side tensor.")
    .add_argument("rhs", "Tensor", "The right hand side tensor.")
    .set_support_level(1)
    .set_attr<FInferShape>("FInferShape", InferShapeBinaryBroadcast)
    .set_attr<FInferType>("FInferType", InferTypeBinaryBroadcast);

}  // namespace relax
}  // namespace tvm