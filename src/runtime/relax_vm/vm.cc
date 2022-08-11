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

#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/packed_func.h>
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

VMFunction VirtualMachine::LookupVMFunction(const std::string& func_name) {
  ICHECK(exec_) << "The executable is not created yet.";
  const auto& m = this->exec_->global_map;
  if (m.find(func_name) == m.end()) {
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }
  Index gf_idx = m.at(func_name);
  const VMFunction& vm_func = exec_->global_funcs[gf_idx];
  return vm_func;
}

RegType VirtualMachine::LookupVMOutput(const std::string& func_name) {
  if (!outputs_.count(func_name)) {
    LOG(FATAL) << "ValueError: No output saved for call of \"" << func_name
               << "\"; use `invoke_stateful` to call it first.";
  }
  return outputs_[func_name];
}

// Use the args after `starting_arg_idx` as a series of indices into `obj`,
// indexing into nested ADTs and returning the final indexed object.
ObjectRef IndexIntoNestedObject(ObjectRef obj, TVMArgs args, int starting_arg_idx) {
  for (int i = starting_arg_idx; i < args.size(); i++) {
    // the object must be an ADT to be able to index into it
    if (!obj.as<ADTObj>()) {
      LOG(FATAL) << "ValueError: Attempted to index into an object that is not an ADT.";
    }
    int index = args[i];
    auto adt = Downcast<ADT>(obj);
    // make sure the index is in bounds
    if (index >= static_cast<int>(adt.size())) {
      LOG(FATAL) << "IndexError: Invalid index (" << index << " >= " << adt.size() << ").";
    }
    obj = adt[index];
  }
  return obj;
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
  } else if (name == "save_function") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      std::string closure_name = args[1];
      bool include_return = args[2];
      const auto& m = exec_->global_map;
      if (m.find(func_name) == m.end()) {
        LOG(FATAL) << "ValueError: Unknown function: " << func_name;
      }
      if (m.find(closure_name) != m.end()) {
        LOG(FATAL) << "ValueError: Name " << closure_name << " is already taken.";
      }
      Index gf_idx = m.at(func_name);
      std::vector<RegType> inputs;
      if (args.size() > 3) {
        inputs = std::vector<RegType>(args.size() - 3);
        for (int i = 3; i < args.size(); i++) {
          inputs[i - 3] = args[i];
        }
      }
      if (include_return) {
        saved_closures_[closure_name] =
            PackedFunc([this, gf_idx, inputs](TVMArgs args, TVMRetValue* rv) {
              *rv = this->Invoke(gf_idx, inputs);
            });
      } else {
        saved_closures_[closure_name] =
            PackedFunc([this, gf_idx, inputs](TVMArgs args, TVMRetValue* rv) {
              this->Invoke(gf_idx, inputs);
            });
      }
    });
  } else if (name == "invoke_closure") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK(exec_) << "The executable is not created yet.";
      VMClosure clo = args[0];
      Array<ObjectRef> func_args = args[1];
      std::vector<RegType> new_args;
      for (auto f_arg : func_args) {
        TVMRetValue arg;
        arg = f_arg;
        new_args.push_back(arg);
      }
      // Append the free variables of closure
      auto free_vars = clo->free_vars;
      for (auto f_var : free_vars) {
        TVMRetValue arg;
        arg = f_var;
        new_args.push_back(arg);
      }

      String func_name = clo->func_name;
      auto it = exec_->global_map.find(func_name);
      ICHECK(it != exec_->global_map.end()) << "No such function " << func_name;
      Index func_idx = it->second;
      *rv = Invoke(func_idx, new_args);
    });
  } else if (name == "invoke_stateful") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      const auto& m = this->exec_->global_map;
      if (m.find(func_name) == m.end()) {
        LOG(FATAL) << "ValueError: Unknown function: " << func_name;
      }
      Index gf_idx = m.at(func_name);
      if (!inputs_.count(func_name)) {
        LOG(FATAL) << "ValueError: No inputs set for stateful call of " << func_name
                   << "; use `set_input` first.";
        return;
      }
      outputs_[func_name] = this->Invoke(gf_idx, inputs_[func_name]);
    });
  } else if (name == "get_output_arity") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      RegType out = LookupVMOutput(func_name);
      // use remaining args as indices
      ObjectRef obj = IndexIntoNestedObject(out.AsObjectRef<ObjectRef>(), args, 1);
      // after chasing through the indices, examine the final object
      if (const auto* adt = obj.as<ADTObj>()) {
        *rv = static_cast<int>(adt->size);
      } else {
        *rv = -1;
      }
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      RegType out = LookupVMOutput(func_name);
      // use remaining args as indices
      ObjectRef obj = IndexIntoNestedObject(out.AsObjectRef<ObjectRef>(), args, 1);
      if (obj.as<ADTObj>()) {
        LOG(FATAL) << "ValueError: `get_output` cannot return a tuple for RPC compatibility. "
                      "Please specify another index argument.";
        return;
      }
      *rv = obj;
    });
  } else if (name == "set_input") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { SetInput(args[0], args, 1); });
  } else if (name == "get_function_arity") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      const VMFunction& vm_func = LookupVMFunction(func_name);
      *rv = static_cast<int>(vm_func.param_names.size());
    });
  } else if (name == "get_function_param_name") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      int index = args[1];
      const VMFunction& vm_func = LookupVMFunction(func_name);
      if (static_cast<size_t>(index) >= vm_func.param_names.size()) {
        LOG(FATAL) << "ValueError: Invalid index for " << func_name << " (" << index << " out of "
                   << vm_func.param_names.size() << ")";
      }
      *rv = vm_func.param_names[index];
    });
  }

  // check if this is a function we saved
  if (saved_closures_.count(name)) {
    return saved_closures_[name];
  }

  const auto& m = exec_->global_map;
  if (m.find(name) != m.end()) {
    Index gf_idx = m.at(name);
    return PackedFunc([sptr_to_self, this, gf_idx, name](TVMArgs args, TVMRetValue* rv) {
      if (inputs_.count(name)) {
        LOG(FATAL) << "ValueError: If inputs have been set, `invoke_stateful`"
                   << " must be used to invoke a function!";
        return;
      } else {
        std::vector<RegType> inputs(args.size());
        for (int i = 0; i < args.size(); ++i) {
          inputs[i] = args[i];
        }
        *rv = this->Invoke(gf_idx, inputs);
      }
    });
  } else {
    LOG(FATAL) << "ValueError: Unknown function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VirtualMachine::LoadExecutable(ObjectPtr<Executable> exec) {
  this->exec_ = exec;
  CHECK_LE(exec_->imports().size(), 1);
  this->lib = exec_->imports().empty() ? Optional<Module>(NullOpt) : exec_->imports()[0];
}

RegType VirtualMachine::Invoke(Index gf_idx, const std::vector<RegType>& args) {
  const VMFunction& gfunc = exec_->global_funcs[gf_idx];
  PushFrame(this->pc_, gfunc);
  // load arguments to the register file
  ICHECK_EQ(static_cast<size_t>(gfunc.num_args), args.size())
      << "ValueError: Invoking function " << gfunc.name << " requires " << gfunc.num_args
      << " inputs but only " << args.size() << " inputs are provided.";
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

  this->devices.reserve(devices.size());
  this->allocators.reserve(alloc_types.size());
  for (size_t i = 0; i < devices.size(); i++) {
    auto alloc = MemoryManager::GetOrCreateAllocator(devices[i], alloc_types[i]);
    this->devices.push_back(devices[i]);
    this->allocators.push_back(alloc);
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
  if (this->lib.defined()) {
    func = this->lib.value()->GetFunction(func_name, true);
  }
  if (!func.defined()) {
    const PackedFunc* p_func = Registry::Get(func_name);
    if (p_func == nullptr) {
      const auto& m = exec_->global_map;
      ICHECK(m.find(func_name) != m.end())
          << "Error: Cannot find function " << func_name
          << " in either Relax VM kernel library, or in TVM runtime PackedFunc registry, or in "
             "global Relax functions of the VM executable";
      func = this->GetFunction(func_name, GetObjectPtr<Object>(this));
    } else {
      func = *(p_func);
    }
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
        if (arg.value() == Instruction::kVMRegister) {
          setter(i, this);
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
        LOG(FATAL) << "ValueError: Unknown argument kind: " << int(arg.kind());
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
        if (frames_.size() == 0) {
          // directly return if no frame in the call stack.
        } else {
          // return from a local call.
          // Update the current frame to be the parent frame.
          curr_frame = frames_.back().get();
          WriteRegister(curr_frame, caller_return_register, return_value_);
        }
        return;
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

void VirtualMachine::SetInput(std::string func_name, TVMArgs args, int offset) {
  const auto& m = exec_->global_map;
  if (m.find(func_name) != m.end()) {
    Index gf_idx = m.at(func_name);
    const VMFunction& vm_func = exec_->global_funcs[gf_idx];
    size_t params_num = vm_func.num_args;
    ICHECK_EQ(args.size() - offset, params_num)
        << "The number of provided parameters doesn't match the number of arguments for";
    std::vector<RegType> func_args(params_num);
    for (int i = offset; i < args.size(); ++i) {
      int index = i - offset;
      SetInputTensorWithIndex(func_args, args[i], index, devices[0]);
    }
    inputs_.emplace(func_name, func_args);
  } else {
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }
}

inline ObjectRef CopyTo(ObjectRef src, const DLDevice& dev) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    auto nd_array = Downcast<NDArray>(src);
    if (nd_array->device.device_type != dev.device_type ||
        nd_array->device.device_id != dev.device_id) {
      VLOG(2) << "copying from " << nd_array->device.device_type << "["
              << nd_array->device.device_id << "] to " << dev.device_type << "[" << dev.device_id
              << "]";
      return nd_array.CopyTo(dev);
    }
    return src;
  } else {
    ICHECK(src->IsInstance<ADTObj>())
        << "VM data must be NDArray or a list of NDArray, but received: " << src->_type_key;
    std::vector<ObjectRef> ret;
    ADT adt = Downcast<ADT>(src);
    for (size_t i = 0; i < adt.size(); i++) {
      ret.push_back(CopyTo(adt[i], dev));
    }
    return ADT(adt->tag, ret.begin(), ret.end());
  }
}

void VirtualMachine::SetInputTensorWithIndex(std::vector<RegType>& func_args,
                                             const TVMArgValue& inp_tensor, int index, Device dev) {
  if (inp_tensor.type_code() == kTVMDLTensorHandle) {
    if (NDArray::AbilityOfZeroCopyForDLTensor(inp_tensor, dev)) {
      func_args[index] = NDArray::FromExternalDLTensor(*inp_tensor);
    } else {
      func_args[index] = NDArray::NewFromDLTensor(inp_tensor, dev);
    }
  } else {
    func_args[index] = CopyTo(inp_tensor, dev);
  }
}

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
