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
 * \brief A codegen to generate VM executable from an IRModule with relax functions.
 */

#ifndef TVM_RELAX_BACKEND_VM_CODEGEN_VM_H_
#define TVM_RELAX_BACKEND_VM_CODEGEN_VM_H_

#include <tvm/ir/module.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
namespace relax {
namespace relax_vm {

using tvm::Target;
using namespace tvm::runtime::relax_vm;
using namespace tvm::runtime;

class VMCodeGen : public Object {
 public:
  /*!
   * \brief Compile the functions in a Module.
   * \param rx_mod Input IRModule that constains relax functions.
   */
  void CodeGen(IRModule rx_mod);
  /*!
   * \brief Get the compiled executable.
   * \return The compiled executable.
   */
  ObjectPtr<Executable> GetExec();

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.VMCodeGen";

 protected:
  /*! \brief Internal executable builder. */
  relax::ExecBuilder builder_;
};

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_VM_CODEGEN_VM_H_
