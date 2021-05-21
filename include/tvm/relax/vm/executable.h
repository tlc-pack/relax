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
 * \file src/relax/vm/executable.h
 * \brief 
 */
#ifndef TVM_RELAX_VM_EXECUTABLE_H_
#define TVM_RELAX_VM_EXECUTABLE_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/reflection.h>
#include <tvm/node/repr_printer.h>
#include <tvm/ir/expr.h>
#include "./bytecode.h"

#include <sstream>

namespace tvm {
namespace relax {
namespace vm {

class Executable;

class ExecutableNode : public Object {
 public:
  std::vector<ExecWord> instr_data;
  // std::vector<ExecWord> arg_data;
  std::vector<size_t> instr_offset;
  std::vector<ObjectRef> constants;
  std::unordered_map<std::string, Index> func2idx;
  std::vector<std::string> func_names;

  // magic number, version,  
  // SaveToBinary(dmlc::Stream* stream);
  // SaveToFile(const std::string& path, const std::string& format);
  
  Instruction GetInstruction(size_t i) const {
    size_t offset = instr_offset[i];
    Opcode op = static_cast<Opcode>(instr_data[offset]);
    switch (op) {
      case Opcode::Call: {
        RegName dst = instr_data[offset + 1];
        Index func_idx = instr_data[offset + 2];
        Index num_args = instr_data[offset + 3];
        const ExecWord* args = &instr_data[offset + 4];
        return Instruction::Call(func_idx, num_args, const_cast<ExecWord*>(args), dst);
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(op);
        break;
    }
    return Instruction();
  }

  String AsText() const;

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Executable"; 
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutableNode, Object);
};

class Executable : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Executable, ObjectRef, ExecutableNode);
};


}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_EXECUTABLE_H_
