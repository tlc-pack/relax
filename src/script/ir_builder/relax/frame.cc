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

void FunctionFrameNode::ExitWithScope() {
  // At this moment, there should be at most one BlockFrame which hasn't ended. In this case, call
  // its `ExitBlockFrame` and check if there is any more unended BlockFrame.
  if (Optional<BlockFrame> block_frame = IRBuilder::Current()->FindFrame<BlockFrame>()) {
    block_frame.value()->ExitWithScope();
    ICHECK(!IRBuilder::Current()->FindFrame<BlockFrame>().defined())
        << "ValueError: There is some remaining BlockFrame that is not properly popped out.";
  }

  using tvm::relax::Expr;
  RelaxFrameNode::ExitWithScope();
  IRBuilder builder = IRBuilder::Current();
  // Step 1: Create the function.
  CHECK(output.defined()) << "ValueError: A Relax function must have a return value. Please use "
                             "`return` to return an Expr";
  output = this->block_builder->Normalize(output.value());
  Expr body = this->block_builder->Normalize(tvm::relax::SeqExpr(binding_blocks, output.value()));
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
    CHECK(name.defined()) << "ValueError: The function name must be defined before exiting the "
                             "function scope, if it's defined in a Module";
    const IRModuleFrame& frame = opt_frame.value();
    const String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name)) {
      // First time visiting the function.
      ir::DeclFunction(func_name);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    ir::DefFunction(func_name, func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert Relax.Function";
  }
}

void BlockFrameNode::EnterWithScope() {
  // Step 1. If the last frame is a block frame. The start of a new block frame marks the end of the
  // last block frame.
  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  if (block_frame.defined()) {
    block_frame.value()->ExitWithScope();
    // Block frames cannot appear consecutively.
    ICHECK(!IRBuilder::Current()->GetLastFrame<BlockFrame>());
  }
  // Step 2. Deal with the new block frame.
  RelaxFrameNode::EnterWithScope();
  Optional<FunctionFrame> func_frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  CHECK(func_frame.defined())
      << "ValueError: Cannot find FunctionFrame when creating BindingBlocks, Please ensure "
         "creating the block under Relax function scope.";
  const tvm::relax::BlockBuilder& block_builder = func_frame.value()->block_builder;
  if (is_dataflow) {
    block_builder->BeginDataflowBlock();
  } else {
    block_builder->BeginBindingBlock();
  }
}

void BlockFrameNode::ExitWithScope() {
  // Step 1. Pop the current frame out of the frame stack.
  RelaxFrameNode::ExitWithScope();

  // Step 2. Get the constructed binding block from the block builder. The block should have at
  // lease one binding - otherwise, the block is not supposed to be created.
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::BindingBlock block = block_builder->EndBlock();
  ICHECK(!block->bindings.empty())
      << "ValueError: A binding block should have at lease one binding.";

  // Step 3. Get the last frame from the IRBuilder frame stack.
  Optional<RelaxFrame> opt_last_frame = IRBuilder::Current()->GetLastFrame<RelaxFrame>();
  ICHECK(opt_last_frame.defined());
  RelaxFrame last_frame = opt_last_frame.value();

  // Step 4. Since we popped out any possible block frame when entering the "with" scope of the
  // current frame, the last frame cannot be a block frame.
  ICHECK(!last_frame->IsInstance<BlockFrameNode>());

  // Step 5. Push the block frame into the corresponding field of the last frame.
  if (const auto* function_frame = last_frame.as<FunctionFrameNode>()) {
    ICHECK(!function_frame->output.defined())
        << "The function is not expected to have output values when emitting blocks.";
    FunctionFrame frame = GetRef<FunctionFrame>(function_frame);
    frame->binding_blocks.push_back(block);
  } else {
    LOG(FATAL) << "ValueError: Currently the last frame is supposed to be either a function frame "
                  "or a block frame. However, the last frame is \""
               << last_frame->GetTypeKey() << "\".";
    // TODO(ruihang): support IfFrame and then IfFrame is a possible branch here.
  }
}

TVM_REGISTER_NODE_TYPE(FunctionFrameNode);
TVM_REGISTER_NODE_TYPE(BlockFrameNode);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
