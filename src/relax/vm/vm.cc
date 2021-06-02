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
  const auto& m = exec_->global_map;
  if (m.find(name) != m.end()) {
    Index gf_idx = m.at(name);
    return PackedFunc([sptr_to_self, this, gf_idx](TVMArgs args, TVMRetValue* rv) {
      std::vector<RegType> inputs(args.size());
      for (int i = 0; i < args.size(); ++i) {
        inputs[i] = args[i];
      }
      *rv = this->Invoke(gf_idx, inputs);
    });
  } else {
    LOG(FATAL) << "Unknown function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VirtualMachine::Load(Executable exec,
                          runtime::Module mod) {
  this->exec_ = exec;
  this->mod_ = mod;
}

RegType VirtualMachine::Invoke(Index gf_idx,
                               const std::vector<RegType>& args) {
  const VMFunction& gfunc = exec_->global_funcs[gf_idx];
  PushFrame(this->pc_ + 1, gfunc);
  // load arguments to the register file
  ICHECK(static_cast<size_t>(gfunc.num_args) == args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  // set program counter
  pc_ = gfunc.start_instr;
  RunLoop();
  return return_value_;
}

void VirtualMachine::RunLoop() {
  size_t start_frame = frames_.size();
  while (true) {
  main_loop:
    if (static_cast<size_t>(pc_) >= exec_->instr_offset.size()) {
      LOG(FATAL) << "run into invalide section";
    }
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        std::string func_name = exec_->func_names[instr.func_idx];
        DLOG(INFO) << "\n  pc = " << pc_ << ", execute: " << func_name;
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
              setter(i, ReadRegister(arg.value()));
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
          WriteRegister(instr.dst, ret);
        }
        pc_++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_value_ = ReadRegister(instr.result);
        auto caller_return_register = frames_.back().caller_return_register;
        PopFrame();
        if (frames_.size() < start_frame) {
          ICHECK(frames_.size() == start_frame - 1);
          return;
        } else {
          // Otherwise we are just returning from a local call.
          WriteRegister(caller_return_register, return_value_);
          goto main_loop;
        }
      }
    }
  }
}

void VirtualMachine::PushFrame(Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, vm_func.register_file_size);
  frames_.push_back(frame);
}

void VirtualMachine::PopFrame() {
  ICHECK_GT(frames_.size(), 0);
  const VMFrame& fr = frames_.back();
  pc_ = fr.return_pc;
  frames_.pop_back();
}

inline void VirtualMachine::WriteRegister(Index r, const RegType& val) {
  frames_.back().register_file[r] = val;
}

inline RegType VirtualMachine::ReadRegister(Index r) const {
  return frames_.back().register_file[r];
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
