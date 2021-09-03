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
 * \file binary.h
 * \brief shape and type deduction for binary broadcast operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/topi/broadcast.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

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

Optional<Expr> InferShapeBinaryBroadcast(const Attrs& attrs, const Call& call) {
  ICHECK_EQ(call->args.size(), 2);
  Expr lhs_shape = call->args[0]->shape();
  Expr rhs_shape = call->args[1]->shape();
  auto* s0 = lhs_shape.as<ShapeExprNode>();
  auto* s1 = rhs_shape.as<ShapeExprNode>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    size_t i = 1;
    for (; i <= std::min(ndim0, ndim1); ++i) {
      PrimExpr dim0 = s0->values[ndim0 - i];
      PrimExpr dim1 = s1->values[ndim1 - i];
      if (EqualConstInt(dim0, 1)) {
        output_shape.push_back(dim0);
      } else if (EqualConstInt(dim1, 1)) {
        output_shape.push_back(dim1);
      } else if (EqualCheck(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else {
        // defer the computation of output shapes to runtime
        // e.g., broadcast Tensor([m, n]), Tensor([k]) -> defer to runtime
        return Call(ExternFunc(String("vm.binary_broadcast_shape_infer")),
                    {call->args[0], call->args[1]}, {}, {});
      }
    }
    size_t max_ndim = std::max(ndim0, ndim1);
    auto& longer_shape = (ndim0 > ndim1) ? s0 : s1;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape->values[max_ndim - i]);
    }
    return ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
  } else {
    return NullOpt;
  }
}

Type InferTypeBinaryBroadcast(const Attrs& attrs, const Call& call) {
  ICHECK_EQ(call->args.size(), 2);
  Type lhs_type = call->args[0]->checked_type();
  Type rhs_type = call->args[1]->checked_type();
  auto* t0 = lhs_type.as<DynTensorTypeNode>();
  auto* t1 = rhs_type.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    LOG(FATAL) << "Both lhs and rhs should be DynTensor for broadcasting";
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype) {
    LOG(FATAL) << "Data types " << t0->dtype << " and " << t1->dtype
               << " do not match broadcasting rule";
  } else {
    output_dtype = t0->dtype;
  }

  int output_rank;
  if (t0->IsUnknownRank() || t1->IsUnknownRank()) {
    output_rank = -1;
  } else {
    output_rank = std::max(t0->rank, t1->rank);
  }
  return DynTensorType(output_rank, output_dtype);
}

}  // namespace relax
}  // namespace tvm
