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

#include <tvm/relax/expr.h>
#include <tvm/script/ir_builder/relax/frame.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

void FunctionFrameNode::EnterWithScope() {
  RelaxFrameNode::EnterWithScope();
  // Start a default binding block scope;
  default_binding_block_frame->EnterWithScope();
}

void FunctionFrameNode::ExitWithScope() {
  using tvm::relax::Expr;
  default_binding_block_frame->ExitWithScope();
  RelaxFrameNode::ExitWithScope();
  IRBuilder builder = IRBuilder::Current();
  // Step 1: Create the function.
  Expr output = outputs.size() == 1 ? outputs[0] : tvm::relax::Tuple(outputs);
  output = this->block_builder->Normalize(output);
  Expr body = this->block_builder->Normalize(tvm::relax::SeqExpr(binding_blocks, output));
  tvm::relax::Function func(/*params=*/params,
                            /*body=*/body,
                            /*ret_type=*/ret_type.value_or(Type()),
                            /*attrs=*/DictAttrs(attrs));
  // TODO(relax-team): remove this line
  func = WithAttr(func, "global_symbol", name.value());
  // Step 2: Update IRModule.
  if (builder->frames.empty()) {
    // Case 0. If there is no output module frame.
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = func;
  } else if (Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    IRModuleFrame frame = opt_frame.value();
    frame->global_vars.push_back(GlobalVar(name.value_or("")));
    frame->functions.push_back(func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert Relax.Function";
  }
}

void BlockFrameNode::EnterWithScope() {
  RelaxFrameNode::EnterWithScope();
  Optional<FunctionFrame> func_frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  CHECK(func_frame.defined()) << "ValueError: Cannot find FunctionFrame when creating "
                                 "BindingBlocks, Please ensure calling "
                              << (is_dataflow ? "R.dataflow()" : "R.block_binding")
                              << " after R.function.";
  const tvm::relax::BlockBuilder& block_builder = func_frame.value()->block_builder;
  if (is_dataflow) {
    block_builder->BeginDataflowBlock();
  } else {
    block_builder->BeginBindingBlock();
  }
}

void BlockFrameNode::ExitWithScope() {
  RelaxFrameNode::ExitWithScope();
  // We've checked that the scope when EnterWithScope, no need to check again.
  FunctionFrame frame = FindFunctionFrame("");
  const tvm::relax::BlockBuilder& block_builder = frame->block_builder;
  tvm::relax::BindingBlock block = block_builder->EndBlock();
  if (!block->bindings.empty()) {
    frame->binding_blocks.push_back(block);
  }
}

TVM_REGISTER_NODE_TYPE(FunctionFrameNode);
TVM_REGISTER_NODE_TYPE(BlockFrameNode);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
