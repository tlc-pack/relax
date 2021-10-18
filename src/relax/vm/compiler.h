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
 * \brief A compiler to compile a relay::Module to the VM executable.
 */

#ifndef TVM_RELAX_VM_COMPILER_H_
#define TVM_RELAX_VM_COMPILER_H_

#include <tvm/target/target.h>
#include <tvm/ir/module.h>
#include <tvm/relax/vm/exec_builder.h>
#include <tvm/relax/vm/executable.h>

#include <string>

namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::Target;

class VMCompiler : public runtime::ModuleNode {
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

  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  const char* type_key() const { return "relax.VMCompiler"; }

 protected:
  /*! \brief Internal executable builder. */
  relax::ExecBuilder builder_;
  /*! \brief Built library. */
  runtime::Module lib_;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RELAX_VM_COMPILER_H_
