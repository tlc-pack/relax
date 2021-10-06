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
 * \file tvm/relax/op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef TVM_RELAX_OP_ATTR_TYPES_H_
#define TVM_RELAX_OP_ATTR_TYPES_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>

#include <string>

namespace tvm {
namespace relax {

using relay::Call;

/*!
 * \brief Infer the output shape for operators. This function will
 * be invoked to fill the \p shape_ field of expressions.
 * \param call The call node.
 * \param diag_ctx The diagnostic context for reporting errors.
 * \return The inferred output shape expression.
 */
using FInferShape =
    runtime::TypedPackedFunc<Optional<RelayExpr>(const Call& call, DiagnosticContext diag_ctx)>;

/*!
 * \brief Infer the output type for operators. This function will
 * be invoked to fill the \p checked_type_ field of expressions.
 * \param call The call node.
 * \param diag_ctx The diagnostic context for reporting errors.
 * \return The inferred output type.
 */
using FInferType = runtime::TypedPackedFunc<Type(const Call& call, DiagnosticContext diag_ctx)>;

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_TYPES_H_
