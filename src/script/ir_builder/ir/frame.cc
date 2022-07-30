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
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/ir_builder/ir/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {

void IRModuleFrameNode::ExitWithScope() {
  Map<GlobalVar, BaseFunc> func_map;
  for (const auto& kv : functions) {
    const GlobalVar& gv = kv.first;
    const Optional<BaseFunc>& func = kv.second;
    CHECK(func.defined()) << "ValueError: function " << gv->name_hint << " is not defined";
    func_map.Set(gv, func.value());
  }
  IRBuilder builder = IRBuilder::Current();
  ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
  builder->result = tvm::IRModule(func_map);
}

TVM_REGISTER_NODE_TYPE(IRModuleFrameNode);

}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
