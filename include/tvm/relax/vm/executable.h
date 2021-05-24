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
  std::vector<ObjectRef> constants;
  std::vector<std::string> func_names;
  std::unordered_map<std::string, Index> func2idx;
  std::vector<size_t> instr_offset;
  std::vector<ExecWord> instr_data;
  
  std::string Stats() const;
  Instruction GetInstruction(size_t i) const;

  TVMByteArray Save();
  static Executable Load(const std::string& code);
  void SaveToBinary(dmlc::Stream* stream);
  static Executable LoadFromBinary(void* stream);
  void SaveToFile(const std::string& path);
  static Executable LoadFromFile(const std::string& file_name);


  String AsText() const;

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Executable"; 
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutableNode, Object);

 private:
  void SaveConstantSection(dmlc::Stream* strm);
  void SaveCodeSection(dmlc::Stream* strm);
  void SavePackedFuncNames(dmlc::Stream* strm);
  void LoadConstantSection(dmlc::Stream* strm);
  void LoadCodeSection(dmlc::Stream* strm);
  void LoadPackedFuncNames(dmlc::Stream* strm);

  /*! \brief The serialized bytecode. */
  std::string code_;
};

class Executable : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Executable, ObjectRef, ExecutableNode);
};


}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_EXECUTABLE_H_
