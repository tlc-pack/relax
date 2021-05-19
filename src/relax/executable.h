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
 * \file src/runtime/vm/executable.h
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

struct Bytecode {
  std::vector<int64_t> text;
  std::vector<int64_t> data;
};

class Executable;

class ExecutableNode : public Object {
 public:
  Bytecode code;

  // default constructor?
  TVM_DLL static Executable Create(Bytecode code);
  // SaveToBinary

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Executable"; 
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutableNode, Object);
 private:
};

class Executable : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Executable, ObjectRef, ExecutableNode);
};


}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
