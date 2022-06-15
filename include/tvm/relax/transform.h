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
 * \file tvm/relax/transform.h
 * \brief Relax specific transformation passes.
 */
#ifndef TVM_RELAX_TRANSFORM_H_
#define TVM_RELAX_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/meta_schedule/apply_history_best.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {
namespace transform {

using Pass = tvm::transform::Pass;
using PassInfo = tvm::transform::PassInfo;
using PassContext = tvm::transform::PassContext;
using Function = tvm::relax::Function;
using DataflowBlock = tvm::relax::DataflowBlock;

/*!
 * \brief Create a function pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*!
 * \brief Create a dataflowblock pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the dataflowblock pass.
 * \param name The name of the dataflowblock pass.
 * \param required The list of the passes that the dataflowblock pass is dependent on.
 *
 * \return The created dataflowblock pass.
 */
TVM_DLL Pass CreateDataflowBlockPass(
    const runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*!
 * \brief Incorrectly transform the dataflow structure as fail testcases.
 *
 * \return The Pass.
 */
TVM_DLL Pass FailTestRewrite();

/*!
 * \brief Perform fused multiply add rewriting in dataflow blocks.
 *
 * \return The Pass.
 */
TVM_DLL Pass FMARewrite();

/*!
 * \brief Perform lambda lifting to lift functions from nested into global.
 *
 * \return The Pass.
 */
TVM_DLL Pass LambdaLift();

/*!
 * \brief Transform all dataflow structure to non-dataflow version.
 *
 * \return The Pass.
 */
TVM_DLL Pass ToNonDataflow();

/*!
 * \brief Perform explicit tensor allocation for call_tir.
 *
 * \return The Pass.
 */
TVM_DLL Pass CallTIRRewrite();

/*!
 * \brief Transform Relax IR to normal form: transform AST to A-normal form, and fill the
 * checked_type_ and shape_ of expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass Normalize();

/*!
 * \brief Apply the best schedule from tuning database.
 *
 * \return The Pass.
 */
TVM_DLL Pass MetaScheduleApplyHistoryBest(const tvm::meta_schedule::Database& database,
                                          Target target);

/*!
 * \brief Bind params of function of the module to constant tensors.
 *
 * \param func_name The name of the function to bind parameters.
 * \param params The parameters to bind.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindParams(String name, Map<String, runtime::NDArray> params);

/*!
 * \brief Fold constant expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass FoldConstant();

/*!
 * \brief Annotate Op Pattern Kind for TIR functions, which is used in FuseOps.
 * \note It is an auto-detect pass for "unscheduled prim_funcs", the op_pattern will be
 *       "opaque" of we can't detect it. Users can manually annotate the attr `op_pattern`
 *       to prim_func.
 * \return The Pass.
 */
TVM_DLL Pass AnnotateTIROpPattern();

/*!
 * \brief This pass groups bindings in a dataflow block of Relax functions and generates a new
 * grouped Relax function for each group, according to the fusion algorithm described in the pass
 * implementation. By grouping bindings into new Relax functions, we substitute the bindings in the
 * function being manipulated into function calls to the new grouped function.
 *
 * A follow-up pass named "FuseTIR" will generate a TIR PrimFunc for each grouped function.
 * \param fuse_opt_level The level of fuse optimization.
 *        -1 indicates that the level will be inferred from pass context.
 * \return The Pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief Fuse relax sub-function into a larger TIR function if possible.
    this pass works together with FuseOps to perform operator fusion.

 * \return The Pass.
 */
TVM_DLL Pass FuseTIR();

/*!
 * \brief Remove unused global relax functions in a IRModule.
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RemoveUnusedFunctions(Array<runtime::String> entry_functions);

/*!
 * \brief Run codegen.
 * \param target_codegens list of codegens
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RunCodegen(Optional<Array<runtime::String>> target_codegens,
                        Array<runtime::String> entry_functions);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_H_
