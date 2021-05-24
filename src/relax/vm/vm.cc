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
 * \file src/relax/vm/vm.cc
 * \brief 
 */

#include <tvm/relax/vm/vm.h>

namespace tvm {
namespace relax {
namespace vm {

class DummyModule : public runtime::ModuleNode {
 public:
  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self) final {
    return nullptr;
  }

  const char* type_key() const final { return "relax.DummyModule"; }
};

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->Run();
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VirtualMachine::Load(Executable exec,
                          runtime::Module mod) {
  this->exec_ = exec;
  this->mod_ = mod;
}

void VirtualMachine::Run() {
  pc_ = 0;
  Index instr_num = exec_->instr_offset.size();
  while (true) {
  main_loop:
    if (pc_ == instr_num) break;
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        std::string func_name = exec_->func_names[instr.func_idx];
        LOG(INFO) << "\n  pc = " << pc_ << ", execute: " << func_name;
        PackedFunc func = mod_->GetFunction(func_name, true);
        if (func == nullptr) {
          func = *(mod_->GetFuncFromEnv(func_name));
        }

        std::vector<TVMValue> values(instr.num_args);
        std::vector<int> tcodes(instr.num_args);
        runtime::TVMArgsSetter setter(values.data(), tcodes.data());
        for (Index i = 0; i < instr.num_args; ++i) {
          InstrArg arg = instr.args[i];
          switch (arg.kind()) {
            case Instruction::kRegister: {
              setter(i, this->register_file[arg.value()]);
              break;
            }
            case Instruction::kImmediate: {
              setter(i, arg.value());
              break;
            }
            case Instruction::kConstIdx: {
              setter(i, this->exec_->constants[arg.value()]);
              break;
            }
            default: {
              LOG(FATAL) << "";
            }
          }
        }
        TVMArgs args(values.data(), tcodes.data(), values.size());
        TVMRetValue ret;
        func.CallPacked(args, &ret);
        if (instr.dst != Instruction::kVoidArg) {
          this->register_file[instr.dst] = ret;
        }
        pc_++;
        goto main_loop;
      }
    }
  }
}

runtime::Module CreateVirtualMachine(Executable exec,
                                     Optional<runtime::Module> mod) {
  runtime::Module mod_;
  if (!mod) {
    mod_ = runtime::Module(make_object<DummyModule>());
  } else {
    mod_ = mod.value();
  }
  auto vm = make_object<VirtualMachine>();
  vm->register_file.resize(100);
  vm->Load(exec, mod_);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("relax.VirtualMachine")
.set_body_typed([](Executable exec, Optional<runtime::Module> mod) {
	return CreateVirtualMachine(exec, mod);
});


}  // namespace vm
}  // namespace relax
}  // namespace tvm
