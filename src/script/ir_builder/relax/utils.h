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
#ifndef TVM_SCRIPT_IR_BUILDER_RELAX_UTILS_H_
#define TVM_SCRIPT_IR_BUILDER_RELAX_UTILS_H_

#include <tvm/script/ir_builder/relax/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

inline FunctionFrame FindFunctionFrame(const String& method) {
  if (Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>()) {
    return frame.value();
  }
  LOG(FATAL) << "ValueError: Function frame not find. Please ensure '" << method
             << "' is called under R.function()";
  throw;
}

inline tvm::relax::BlockBuilder GetBlockBuilder() {
  Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  CHECK(frame.defined()) << "ValueError: Relax Function frame not find. Please ensure "
                            "assignment is called under R.function()";
  return frame.value()->block_builder;
}

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_UTILS_H_
