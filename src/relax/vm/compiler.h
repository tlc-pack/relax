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
 * \file src/relax/vm/compiler.h
 * \brief A compiler from relay::Module to the VM byte code.
 */

#ifndef TVM_RELAX_VM_COMPILER_H_
#define TVM_RELAX_VM_COMPILER_H_

#include <tvm/relax/vm/exec_builder.h>
#include <tvm/relax/vm/executable.h>
#include <tvm/relax/vm/memory_manager.h>
#include <tvm/relax/vm/vm.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

struct VMCompilerContext {};

class VMCompiler : public runtime::ModuleNode {
 public:
  virtual ~VMCompiler() {}

  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  const char* type_key() const { return "VMCompiler"; }

  void SetParam(const std::string& name, runtime::NDArray data_in);

  void Lower();

 protected:
  /*!
   * \brief Populate the global function names in a map where the value is used
   *        as the index by the VMFunctions.
   */
  void PopulateGlobalMap();

 protected:
  /*! \brief Global shared meta data */
  VMCompilerContext context_;
  /*! \brief Compiled executable. */
  ObjectPtr<Executable> exec_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RELAX_VM_COMPILER_H_
