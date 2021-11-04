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
 * \file src/relax/backend/vm/codegen_vm.h
 * \brief A compiler to compile an IRModule to VM executable.
 */

#ifndef TVM_RELAX_BACKEND_VM_COMPILER_H_
#define TVM_RELAX_BACKEND_VM_COMPILER_H_

#include <tvm/ir/module.h>
#include <tvm/relax/vm/exec_builder.h>
#include <tvm/relax/vm/executable.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
namespace relax {
namespace relax_vm {

using tvm::Target;
using namespace tvm::runtime::relax_vm;
using namespace tvm::runtime;

class VMCompiler : public Object {
 public:
  /*!
   * \brief Compile the functions in a Module.
   * \param mod Input IRModule to be compiled.
   */
  void Compile(IRModule mod, Target target, Target target_host);
  /*!
   * \brief Get the compiled executable.
   * \return The compiled executable.
   */
  Executable GetExec();
  /*!
   * \brief Get the compiled library.
   * \return The compiled lirary.
   */
  Module GetLib();

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.VMCompiler";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutableNode, Object);

 protected:
  /*! \brief Internal executable builder. */
  relax::ExecBuilder builder_;
  /*! \brief Built library. */
  runtime::Module lib_;
};

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_VM_COMPILER_H_
