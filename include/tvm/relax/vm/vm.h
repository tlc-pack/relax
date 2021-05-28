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
 * \file src/relax/vm/vm.h
 * \brief 
 */
#ifndef TVM_RELAX_VM_VM_H_
#define TVM_RELAX_VM_VM_H_

#include "./bytecode.h"
#include "./executable.h"

namespace tvm {
namespace relax {
namespace vm {

struct VMFrame {
  Index return_pc;
  std::vector<ObjectRef> register_file;
  RegName caller_return_register;

  VMFrame(Index pc, Index register_file_size)
      : return_pc(pc),
        register_file(register_file_size),
        caller_return_register(0) {}
};


class VirtualMachine : public runtime::ModuleNode {
 public:
  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self) final;

  virtual ~VirtualMachine() final {}

  const char* type_key() const final { return "relax.VirtualMachine"; }

  void Load(Executable exec, runtime::Module mod);

 protected:

  void PushFrame(Index ret_pc, const VMFunction& vm_func);

  void PopFrame();

  inline void WriteRegister(RegName reg, const ObjectRef& obj);

  inline ObjectRef ReadRegister(RegName reg) const;

  ObjectRef Invoke(Index fidx, const std::vector<ObjectRef>& args);

  void RunLoop();

 private:
  Executable exec_;
  runtime::Module mod_;
  std::vector<VMFrame> frames_;
  Index pc_{0};
  ObjectRef return_value_;
};

}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_VM_H_
