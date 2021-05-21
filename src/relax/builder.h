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
 * \file tvm/runtime/vm/bytecode.h
 * \brief 
 */
#ifndef TVM_RUNTIME_NEW_VM_BUILDER_H_
#define TVM_RUNTIME_NEW_VM_BUILDER_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/reflection.h>
#include <tvm/node/repr_printer.h>
#include <tvm/ir/expr.h>
#include "./bytecode.h"
#include "./executable.h"

namespace tvm {
namespace runtime {
namespace new_vm {

class Builder;

class BuilderNode : public Object {
 public:
  ObjectPtr<ExecutableNode> exec; // mutable

  void EmitCall(std::string func, std::vector<InstrArg> args, RegName ret);

  Index AddConstant(ObjectRef obj) {
    Index idx = exec->constants.size();
    exec->constants.push_back(obj);
    return InstrArg(kConstIdx, idx).data;
  }

  Executable Get();

  void Print(std::ostream& os);

  TVM_DLL static Builder Create();

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Builder"; 
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderNode, Object);
};

class Builder : public ObjectRef {
 public:
  // Builder
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Builder, ObjectRef, BuilderNode);
};


}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NEW_VM_BUILDER_H_
