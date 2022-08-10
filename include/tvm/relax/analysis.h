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
#include <tvm/relax/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/function.h>

#include <utility>

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

/*!
 * \brief Get all bound variables from expression expr.
 *
 * Bound variables are all variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> BoundVars(const Expr& expr);

/*!
 * \brief Get free type parameters from expression expr.
 *
 * Free variables are variables that are not bound by a
 * varbinding or a function parameter in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> FreeVars(const Expr& expr);

/*!
 * \brief Get all variables from expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> AllVars(const Expr& expr);

/*!
 * \brief Get all glabal variables for recursive call from expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all global variables for recursive call.
 */
TVM_DLL tvm::Array<GlobalVar> RecGlobalVars(const Expr& expr);

/*!
 * \brief Get all glabal variables from expression expr.
 *
 * AllVars is a superset of BoundVars and FreeVars.
 * The union of BoundVars and FreeVars is Allvars.
 *
 * \param expr the expression.
 *
 * \return List of all global variables, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<GlobalVar> AllGlobalVars(const Expr& expr);

/*!
 * \brief Analyze var -> value mapping from VarBindings.
 *
 * \param m The IRModule to check.
 * \return Var -> Value (Expr)
 */
TVM_DLL Map<Var, Expr> AnalyzeVar2Value(const IRModule& m);

/*!
 * \brief Analyze var -> value mapping from VarBindings.
 *
 * \param expr The expression to check.
 * \return Var -> Value (Expr)
 */
TVM_DLL Map<Var, Expr> AnalyzeVar2Value(const Expr& expr);

/*!
 * \brief Analyze var -> value mapping from VarBindings.
 *
 * \param dfb The dataflow block to check.
 * \return Var -> Value (Expr)
 */
TVM_DLL Map<Var, Expr> AnalyzeVar2Value(const DataflowBlock& dfb);

/*!
 * \brief Return a mapping from variable name to its Bindings.
 *
 * \param fn The function to be analyzed.
 * \return A mapping from variable name to its Bindings.
 */
TVM_DLL Map<String, Array<Binding>> NameToBinding(const Function& fn);

/*!
 * \brief Get the use-def chain of variables inside a dataflow block.
 *
 * \param dfb The dataflow block to be analyzed.
 * \return A map mapping variable definitoins to a set of uses.
 */
TVM_DLL Map<Var, Array<Var>> DataflowBlockUseDef(const DataflowBlock& dfb);

/*!
 * \brief Get the use-def chain of variables inside a function.
 *
 * \param fn The function to be analyzed.
 * \return A map from variable definitoins to a set of uses and variables needed by return value.
 */
std::pair<Map<Var, Array<Var>>, Array<Var>> FunctionUseDef(const Function& fn);

/*!
 * \brief Remove unused statements inside DataflowBlocks.
 *
 * \param fn The function to remove unused statements.
 * \return The function that contains no unused statements in DataflowBlock.
 */
TVM_DLL Function RemoveAllUnused(const Function fn);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_H_
