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
 * \file tvm/relax/analysis.h
 * \brief The set of Relax specific analysis passes.
 */
#ifndef TVM_RELAX_ANALYSIS_H_
#define TVM_RELAX_ANALYSIS_H_

#include <tvm/ir/diagnostic.h>
#include <tvm/ir/module.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

/*!
 * \brief Check if the IRModule is well formed.
 *
 * \param m the IRModule to check.
 * \param diag_ctx the diagnostic context.
 * \return true if the IRModule is well formed, false if not.
 */
TVM_DLL bool WellFormed(const IRModule& m,
                        Optional<DiagnosticContext> diag_ctx = Optional<DiagnosticContext>());

/*!
 * \brief Annotate Op Pattern Kind for PrimFunc, which is used in relax FuseOps.
 *
 * \param func The PrimFunc to be analyzed.
 * \return The Op Pattern Kind.
 *
 * \note This analysis applies on TIR function but is primarily used by relax passes.
 *       As a result we place it under the relax namespace.
 */
TVM_DLL relay::OpPatternKind AnalyzeOpPatternKind(const tir::PrimFunc& func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_H_
