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
 * \file src/runtime/relax_vm/vm.cc
 */

#include <tvm/runtime/relax_vm/vm.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

inline TVMRetValue CopyConstantTo(TVMRetValue src, const DLDevice& dev) {
  NDArray nd_array = src.operator tvm::runtime::NDArray();
  if (nd_array->device.device_type == dev.device_type &&
      nd_array->device.device_id == dev.device_id) {
    return src;
  }
  TVMRetValue ret;
  ret = nd_array.CopyTo(dev);
  return ret;
}

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "vm_initialization") {
    // initialize the VirtualMachine, takes variable-length arguments
    // first argument is a runtime::Module, followed by one or more device_type, device_id,
    // and the AllocatorType associated with the device.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size() % 3, 0);
      std::vector<Device> devices;
      std::vector<AllocatorType> alloc_types;
      for (int i = 0; i < args.size(); i += 3) {
        Device dev;
        int device_type = args[i];
        dev.device_type = DLDeviceType(device_type);
        dev.device_id = args[i + 1];
        int type = args[i + 2];
        devices.push_back(dev);
        alloc_types.push_back(AllocatorType(type));
      }
      this->Init(devices, alloc_types);

      // Copy NDArray constants to the devices
      // TODO(tvm-team): support multiple devices
      this->constants.reserve(exec_->constants.size());
      for (const auto& constant : exec_->constants) {
        if (constant.type_code() != kTVMNDArrayHandle) {
          this->constants.push_back(constant);
        } else {
          this->constants.push_back(CopyConstantTo(constant, devices[0]));
        }
      }
    });
  }

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

void VirtualMachine::LoadExecutable(ObjectPtr<Executable> exec) {
  this->exec_ = exec;
  CHECK_LE(exec_->imports().size(), 1);
  this->state.lib = exec_->imports().empty() ? Optional<Module>(NullOpt) : exec_->imports()[0];
}

RegType VirtualMachine::Invoke(Index gf_idx, const std::vector<RegType>& args) {
  const VMFunction& gfunc = exec_->global_funcs[gf_idx];
  PushFrame(this->pc_ + 1, gfunc);
  // load arguments to the register file
  ICHECK(static_cast<size_t>(gfunc.num_args) == args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(frames_.back().get(), i, args[i]);
  }
  // set program counter
  pc_ = gfunc.start_instr;
  RunLoop();
  return return_value_;
}

void VirtualMachine::Init(const std::vector<Device>& devices,
                          const std::vector<AllocatorType>& alloc_types) {
  // TODO(@yuchen): support multi-device heterogeneous execution
  ICHECK_LT(devices.size(), 3)
      << "Currently relax vm only supports at most 2 devices (host + device)";
  ICHECK_EQ(devices.size(), alloc_types.size());

  state.devices.reserve(devices.size());
  state.allocators.reserve(alloc_types.size());
  for (size_t i = 0; i < devices.size(); i++) {
    auto alloc = MemoryManager::GetOrCreateAllocator(devices[i], alloc_types[i]);
    state.devices.push_back(devices[i]);
    state.allocators.push_back(alloc);
  }
}

void VirtualMachine::PrepareFuncTable(Index func_index) {
  // fast path, function already in cache;

  if (static_cast<Index>(func_table_.size()) > func_index && func_table_[func_index] != nullptr)
    return;

  if (static_cast<Index>(func_table_.size()) <= func_index) {
    func_table_.resize(func_index + 1, nullptr);
  }

  const std::string& func_name = exec_->func_names[func_index];
  // lookup function and populate
  PackedFunc func{nullptr};
  if (state.lib.defined()) {
    func = state.lib.value()->GetFunction(func_name, true);
  }
  if (!func.defined()) {
    const PackedFunc* p_func = Registry::Get(func_name);
    CHECK(p_func != nullptr);
    func = *(p_func);
  }
  func_table_[func_index] = func;
}

void VirtualMachine::RunInstrCall(VMFrame* curr_frame, Instruction instr) {
  DLOG(INFO) << "\n  pc = " << pc_ << ", execute: " << exec_->func_names[instr.func_idx];

  // Use the call arg stack from the current frame to increase reuse
  // and avoid re-allocation
  curr_frame->call_arg_values.resize(instr.num_args);
  curr_frame->call_arg_tcodes.resize(instr.num_args);

  // NOTE: no changes and resize to those vector ref(otherwise can leads to segfault)
  //       in the remainder part of the function.
  std::vector<TVMValue>& values = curr_frame->call_arg_values;
  std::vector<int>& tcodes = curr_frame->call_arg_tcodes;

  runtime::TVMArgsSetter setter(values.data(), tcodes.data());
  for (Index i = 0; i < instr.num_args; ++i) {
    Instruction::Arg arg = instr.args[i];
    switch (arg.kind()) {
      case Instruction::kRegister: {
        if (arg.value() == Instruction::kVMStateRegister) {
          setter(i, &(this->state));
        } else {
          setter(i, ReadRegister(curr_frame, arg.value()));
        }
        break;
      }
      case Instruction::kImmediate: {
        setter(i, arg.value());
        break;
      }
      case Instruction::kConstIdx: {
        setter(i, this->constants[arg.value()]);
        break;
      }
      default: {
        LOG(FATAL) << "";
      }
    }
  }
  TVMArgs args(values.data(), tcodes.data(), values.size());
  TVMRetValue ret;
  // prepare and invoke
  this->PrepareFuncTable(instr.func_idx);
  func_table_[instr.func_idx].CallPacked(args, &ret);

  if (instr.dst != Instruction::kVoidArg) {
    WriteRegister(curr_frame, instr.dst, ret);
  }
  pc_++;
}

void VirtualMachine::RunLoop() {
  size_t start_frame = frames_.size();
  VMFrame* curr_frame = frames_.back().get();

  while (true) {
    ICHECK_LT(static_cast<size_t>(pc_), exec_->instr_offset.size()) << "run into invalide section";
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        this->RunInstrCall(curr_frame, instr);
        break;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_value_ = ReadRegister(curr_frame, instr.result);
        RegName caller_return_register = curr_frame->caller_return_register;
        PopFrame();
        if (frames_.size() < start_frame) {
          ICHECK(frames_.size() == start_frame - 1);
          return;
        } else {
          // Update the current frame to be the parent frame.
          curr_frame = frames_.back().get();
          // Otherwise we are just returning from a local call.
          WriteRegister(curr_frame, caller_return_register, return_value_);
        }
        break;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t cond_val = ReadRegister(curr_frame, instr.cond);
        if (cond_val != 0) {
          pc_++;
        } else {
          ICHECK_GT(instr.false_offset, 1);
          pc_ += instr.false_offset;
        }
        break;
      }
    }
  }
}

void VirtualMachine::PushFrame(Index ret_pc, const VMFunction& vm_func) {
  frames_.emplace_back(std::make_unique<VMFrame>(ret_pc, vm_func.register_file_size));
}

void VirtualMachine::PopFrame() {
  ICHECK_GT(frames_.size(), 0);
  pc_ = frames_.back()->return_pc;
  frames_.pop_back();
}

inline void VirtualMachine::WriteRegister(VMFrame* frame, Index r, const RegType& val) {
  frame->register_file[r] = val;
}

inline RegType VirtualMachine::ReadRegister(VMFrame* frame, Index r) const {
  return frame->register_file[r];
}

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
