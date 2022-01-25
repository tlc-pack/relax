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
namespace runtime {
namespace relax_vm {

class DummyModule : public runtime::ModuleNode {
 public:
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
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

void VirtualMachine::Load(Executable exec, runtime::Module mod) {
  this->exec_ = exec;
  this->state.mod_ = mod;
}

RegType VirtualMachine::Invoke(Index gf_idx, const std::vector<RegType>& args) {
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

void VirtualMachine::Init(const std::vector<Device>& devices,
                          const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(devices.size(), alloc_types.size());
  for (size_t i = 0; i < devices.size(); i++) {
    auto dev_type = static_cast<size_t>(devices[i].device_type);
    auto alloc = MemoryManager::GetOrCreateAllocator(devices[i], alloc_types[i]);
    if (devices_.size() <= dev_type) {
      devices_.resize(dev_type + 1);
      state.allocators.resize(dev_type + 1);
    }
    devices_[dev_type] = devices[i];
    state.allocators[dev_type] = alloc;
  }
}

void VirtualMachine::RunLoop() {
  size_t start_frame = frames_.size();
  while (true) {
    if (static_cast<size_t>(pc_) >= exec_->instr_offset.size()) {
      LOG(FATAL) << "run into invalide section";
    }
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        std::string func_name = exec_->func_names[instr.func_idx];
        DLOG(INFO) << "\n  pc = " << pc_ << ", execute: " << func_name;
        PackedFunc func = state.mod_->GetFunction(func_name, true);
        if (func == nullptr) {
          func = *(state.mod_->GetFuncFromEnv(func_name));
        }

        std::vector<TVMValue> values(instr.num_args);
        std::vector<int> tcodes(instr.num_args);
        runtime::TVMArgsSetter setter(values.data(), tcodes.data());
        for (Index i = 0; i < instr.num_args; ++i) {
          Instruction::Arg arg = instr.args[i];
          switch (arg.kind()) {
            case Instruction::kRegister: {
              if (arg.value() == Instruction::kVMStateRegister) {
                setter(i, &(this->state));
              } else {
                setter(i, ReadRegister(arg.value()));
              }
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
        break;
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
        }
        // Otherwise we are just returning from a local call.
        WriteRegister(caller_return_register, return_value_);
        break;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t test_val = ReadRegister(instr.test);
        int64_t target_val = ReadRegister(instr.target);
        if (test_val == target_val) {
          ICHECK_NE(instr.true_offset, 0);
          pc_ += instr.true_offset;
        } else {
          ICHECK_NE(instr.false_offset, 0);
          pc_ += instr.false_offset;
        }
        break;
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

runtime::Module CreateVirtualMachine(Executable exec, Optional<runtime::Module> mod) {
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

// initialize the VirtualMachine, takes variable-length arguments
// first argument is a runtime::Module, followed by one or more device_type, device_id,
// and the AllocatorType associated with the device.
TVM_REGISTER_GLOBAL("relax.VirtualMachineInit").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_EQ(args.size() % 3, 1);
  runtime::Module mod = args[0];
  auto vm = static_cast<VirtualMachine*>(mod.operator->());
  std::vector<Device> devices;
  std::vector<AllocatorType> alloc_types;
  for (int i = 0; i < args.size() / 3; ++i) {
    Device dev;
    int device_type = args[i * 3 + 1];
    dev.device_type = DLDeviceType(device_type);
    dev.device_id = args[i * 3 + 2];
    int type = args[i * 3 + 3];
    devices.push_back(dev);
    alloc_types.push_back(AllocatorType(type));
  }
  vm->Init(devices, alloc_types);
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
