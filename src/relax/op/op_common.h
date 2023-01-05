/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for Relax ops.
 */
#ifndef TVM_RELAX_OP_OP_COMMON_H_
#define TVM_RELAX_OP_OP_COMMON_H_

#include <tvm/arith/analyzer.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/************ Op input struct info getter ************/

/*!
 * \brief Get the tensor struct info of the operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct info of each input.
 * \note This function require every input to be Tensor. The number of call arguments is required
 * to match the number of inputs of the op being called.
 */
Array<TensorStructInfo> GetInputTensorStructInfo(const Call& call, const BlockBuilder& ctx);

/*!
 * \brief Get the tensor struct info of the unary operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct info of the unary operator input.
 * \throw Throw exception if the number of input is not one, or the struct info of the input is not
 * a tensor struct info.
 */
inline TensorStructInfo GetUnaryInputTensorStructInfo(const Call& call, const BlockBuilder& ctx) {
  return GetInputTensorStructInfo(call, ctx)[0];
}

/************ Op registration macro ************/

/*!
 * \brief Quick helper macro to
 * - expose a make-function interface which construct the call node.
 * - register op to the registry.
 * \param OpName The name of operator to register.
 * \param RequireFloatDtype A boolean indicating if the input is required to have float dtype.
 */
#define RELAX_REGISTER_UNARY_OP_INTERFACE(OpName, RequireFloatDtype) \
  Expr OpName(Expr x) {                                              \
    static const Op& op = Op::Get("relax." #OpName);                 \
    return Call(op, {std::move(x)}, Attrs(), {});                    \
  }                                                                  \
  TVM_REGISTER_GLOBAL("relax.op." #OpName).set_body_typed(OpName);   \
  TVM_REGISTER_OP("relax." #OpName)                                  \
      .set_num_inputs(1)                                             \
      .add_argument("x", "Tensor", "The input tensor.")              \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnary<RequireFloatDtype>)

template <bool require_float_dtype>
inline StructInfo InferStructInfoUnary(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  if (require_float_dtype && !input_sinfo->IsUnknownDtype() && !input_sinfo->dtype.is_float()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << call->op
        << " requires the input tensor to have float dtype. However, the given input dtype is "
        << input_sinfo->dtype);
  }
  return input_sinfo;
}

/************ Utilities ************/

/*!
 * \brief Infer the output datatype for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param x1_sinfo The struct info of the first operand
 * \param x2_sinfo The struct info of the second operand
 * \return The inferred output dtype.
 * \throw Throw exception if the dtype of two input TensorStructInfo don’t match
 */
inline DataType InferBinaryArithOpOutDtype(const Call& call, const BlockBuilder& ctx,
                                           const TensorStructInfo& x1_sinfo,
                                           const TensorStructInfo& x2_sinfo) {
  if (x1_sinfo->IsUnknownDtype() || x2_sinfo->IsUnknownDtype()) {
    return DataType::Void();
  } else if (x1_sinfo->dtype != x2_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << x1_sinfo->dtype << " and " << x2_sinfo->dtype
                     << " must be equal for binary operators");
  }
  return x1_sinfo->dtype;
}

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param x1_shape The shape of the first operand.
 * \param x2_shape The shape of the second operand.
 * \return The inferred output shape after broadcasting. Or `NullOpt` if the output shape cannot be
 * determined due to symbolic broadcast.
 */
Optional<Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call, const BlockBuilder& ctx,
                                                    const Array<PrimExpr>& x1_shape,
                                                    const Array<PrimExpr>& x2_shape);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_COMMON_H_
