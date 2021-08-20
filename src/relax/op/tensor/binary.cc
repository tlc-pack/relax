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

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/op.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

using Expr = tvm::RelayExpr;
using relay::Call;

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
    *rv = ShapeTuple(output_shape.rbegin(), output_shape.rend());
  }
}

Optional<Expr> InferShapeBinaryBroadcast(Call call) {
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
        // e.g., broadcast Tensor([m, n]), Tensor([k]) -> defer to runtime
        return PackedFunc(BinaryBroadcast)(lhs_shape, rhs_shape);
        // return Call()
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

Type InferTypeBinaryBroadcast(Call call) {
  ICHECK_EQ(call->args.size(), 2);
  Type lhs_type = call->args[0]->checked_type_;
  Type rhs_type = call->args[1]->checked_type_;
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

RELAX_REGISTER_BINARY_OP("add").describe("Elementwise add with broadcasting").set_support_level(1);

RELAX_REGISTER_BINARY_OP("multiply")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1);

}  // namespace relax
}  // namespace tvm