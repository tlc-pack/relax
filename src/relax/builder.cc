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
 * \file src/runtime/vm/builder.ccregistry
 */

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/reflection.h>
#include <tvm/node/repr_printer.h>
#include <tvm/ir/expr.h>
#include "./bytecode.h"

#include <sstream>

namespace tvm {
namespace runtime {
namespace new_vm {


class Builder;

class BuilderNode : public Object {
 public:
  std::vector<Instruction> code;
  std::unordered_map<std::string, Index> func_idx;

  void Emit(Instruction instr) {}

  void EmitCall(std::string func, std::vector<InstrArg> args, RegName ret) {
    if (func_idx.find(func) == func_idx.end()) {
      func_idx[func] = func_count_++;
    } 
    Instruction instr = Instruction::Call(func_idx[func], args.size(), args, ret);
    LOG(INFO) << "\n" << instr;
    code.push_back(instr);
  }

  // Executable Get() {}
  TVM_DLL static Builder Create();

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Builder"; 
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderNode, Object);
 private:
  Index func_count_{0};
};

class Builder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Builder, ObjectRef, BuilderNode);
};

Builder BuilderNode::Create() {
  Builder ret(make_object<BuilderNode>());
  return ret;
}

TVM_REGISTER_NODE_TYPE(BuilderNode);

TVM_REGISTER_GLOBAL("relax.BuilderCreate").set_body_typed(BuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.BuilderEmitCall").set_body_typed(
[](Builder builder, String name, Array<IntImm> args, int64_t ret) {
  LOG(INFO) << "cpp builder emit call";
  std::vector<InstrArg> args_;
  for (size_t i = 0; i < args.size(); ++i) {
    args_.push_back(static_cast<InstrArg>(args[i]->value));
  }
  InstrArg ret_(ret);
  CHECK_EQ(ret_.kind(), ArgKind::kRegister);
  builder->EmitCall(name, args_, ret_.value());
});

}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
