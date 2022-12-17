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
 * \brief The set of Relax specific analysis on IR.
 */
#ifndef TVM_RELAX_ANALYSIS_H_
#define TVM_RELAX_ANALYSIS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/diagnostic.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/function.h>

#include <functional>
#include <utility>

namespace tvm {
namespace relax {
//-----------------------------------
// Shape expression analysis
//----------------------------------
/*!
 * \brief Can prove the two symbolic shape arrays equals to each other.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana The analyzer used for integer analysis.
 * \return The prove result.
 *
 * \note This function does best effort prove, which means
 *       if result is false, there is still possibility that
 *       two shapes equals to each other during runtime.
 */
TVM_DLL bool CanProveShapeEqual(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs,
                                arith::Analyzer* ana);

/*!
 * \brief Can prove the two symbolic shape expressions equals to each other.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana The analyzer used for integer analysis.
 *
 * \note This function does best effort prove, which means
 *       if result is false, there is still possibility that
 *       two shapes equals to each other during runtime.
 */
TVM_DLL bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs, arith::Analyzer* ana);

//-----------------------------------
// Foundational StructInfo analysis
//-----------------------------------
/*!
 * \brief Get the corresponding static type from a given struct info.
 * \param info The struct info.
 * \return the corresponding static type.
 */
TVM_DLL Type GetStaticType(const StructInfo& info);

/*!
 * \brief Get the corresponding struct info from static type.
 * \param type The input type
 * \return the corresponding struct info.
 */
TVM_DLL StructInfo StructInfoFromType(const Type& type);

// TODO(relax-team): Remove legacy shape related functionalities after phasing out shape_
/*!
 * \brief Get the corresponding struct info from static type.
 * \param type The input type
 * \param shape_hint The shape hint
 * \return the corresponding struct info.
 */
TVM_DLL StructInfo StructInfoFromTypeLegacyShapeHint(const Type& type, Optional<Expr> shape_hint);

/*!
 * \brief Get the corresponding legacy shape hint from struct info
 * \param info The struct info.
 * \return the corresponding legacy shape hint.
 */
TVM_DLL Optional<Expr> GetLegacyShapeHint(const StructInfo& info);

/*!
 * \brief Erase the info to a corresponding more coarse grained
 *        struct info that is still well-defined(with all the vars in scope).
 *
 * When we are returning a StructInfo to another scope,
 * it is important to remember that StructInfo may carry
 * dependencies on var that is not defined the other scope.
 *
 * In such cases, it is important to call EraseToWellDefined to get
 * another StructInfo that **only** contains the vars that are defined
 * in the target scope.
 *
 * For example, consider the following function
 *
 * \code
 *
 * @R.function
 * def f(x: R.Tensor[(n, m)]):
 *     k = tir.Var("k", "int64")
 *     v0 = opaque_fn(x)
 *     v1 = match_cast(v0, R.Tensor[(n, k)])
 *     v2 : R.Tensor[(n + 1, k + 2)] = pad(v1)
 *     return v2
 *
 * \endcode
 *
 * In the above code, the return value y have shape `(n + 1, k + 2)`,
 * However, at the level of function signature, only n, m are defined,
 * k is undefined here.
 *
 * When we call EraseToWellDefined(R.Tensor[(n + 1, k + 2)], fshape_var_map={n: n, m: m}),
 * we will obtain R.Tensor(ndim=2), which is an erased info that does not depend
 * on k(which is undefined from parameter signature).
 *
 * However, if we call EraseToWellDefined(R.Tensor[(n + 1, m)], fshape_var_map={n: n, m: m}),
 * Then the return value will be R.Tensor[(n + 1, m)], because both n and m are defined.
 *
 * We can also make these var map to return a different expression.
 * For example, EraseToWellDefined(R.Tensor[(n + 1, m)], fshape_var_map={n: 2, m: m})
 * will give us R.Tensor[(3, m)], where n get replaced by 2.
 *
 * Use this function in the following scenarios:
 * - Decide the struct_info of expr with sub-scopes, such as If, SeqExpr
 * - Decide the deduced return struct_info of a function that can be fully decided by params.
 *
 * \param info The struct info.
 * \param f_shape_var_map callback function to specify
 *        whether a symbolic shape var is defined and the value it maps to,
 *        return nullopt if var is undefined.
 * \param f_var_defined callback function to specify
 *        whether a var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 *
 * \return the corresponding erased struct info.
 */
TVM_DLL StructInfo
EraseToWellDefined(const StructInfo& info,
                   std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map = nullptr,
                   std::function<Optional<Expr>(const Var& var)> f_var_map = nullptr,
                   arith::Analyzer* ana = nullptr);

/*!
 * \brief Check the relation of two struct info to see if one subsumes another one.
 * \param base The base struct info.
 * \param derived The derived struct info.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 * \return Whether the relation holds.
 */
TVM_DLL bool IsBaseOf(const StructInfo& base, const StructInfo& derived,
                      arith::Analyzer* ana = nullptr);

/*!
 * \brief Unify the two struct info their least common ancestor.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 * \return The unified information.
 */
TVM_DLL StructInfo StructInfoLCA(const StructInfo& lhs, const StructInfo& rhs,
                                 arith::Analyzer* ana = nullptr);

//-----------------------------------
// General IR analysis
//----------------------------------
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
 * \brief Gather all shape variables from expression expr.
 *
 * This analysis is intended to be called on shape expressions (those set as the shape_ of another
 * expression).
 *
 * \param expr the expression. Meant to be a shape expression.
 *
 * \return List of shape variables (tir::Var)
 */
TVM_DLL tvm::Array<tir::Var> ShapeVars(const Expr& expr);

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
 * \brief Get all glabal variables used in calls in expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all global variables called in expr.
 */
TVM_DLL tvm::Array<GlobalVar> CalledGlobalVars(const Expr& expr);

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

/*!
 * \brief Given the argument vars and body, derives a return shape for a function with those args
 * and that body. If the body's shape contains free shape vars (those not used in the args), the
 * return shape is relaxed to RuntimeDepShape; otherwise, the body's shape is used.
 *
 * \param args The argument variables, ideally with the shape_ field filled in
 * \param body The functino body, ideally with the shape_ field filled in
 * \return An expression that can serve as the return shape for the function
 */
TVM_DLL Expr DeriveFuncRetShape(Array<Var> args, Expr body);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_H_
