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
 * \file src/runtime/vm/vm.cc
 * \brief The virtual machine runtime.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "./vm.h"
#include "../runtime/file_utils.h"

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace new_vm {

void VMFunctionPrint(std::ostream& os, const VMFunction& vm_func) {
  os << vm_func.name << ": " << std::endl;
  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    os << i << ": " << vm_func.instructions[i] << ";" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, const VMFunction& vm_func) {
  VMFunctionPrint(os, vm_func);
  return os;
}

inline ObjectRef CopyTo(ObjectRef src, const DLDevice& dev) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    auto nd_array = Downcast<NDArray>(src);
    if (nd_array->device.device_type != dev.device_type) {
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

std::vector<int64_t> ToShape(NDArray shape_tensor) {
  std::vector<int64_t> shape;
  auto rank = shape_tensor.Shape().size();
  auto dtype = shape_tensor.DataType();

  // For 0-rank shapes we need to allocate a single scalar.
  if (rank == 0) {
    return shape;
  }

  // Otherwise we should be rank-1, and we will extract the number of dimensions
  // for the output vector.
  ICHECK_EQ(rank, 1U) << "shape tensor should be a k-length vector, found " << rank;
  int64_t ndim = shape_tensor.Shape().at(0);
  shape.resize(ndim);

  const DLTensor* dl_tensor = shape_tensor.operator->();
  if (dtype.is_int() && dtype.bits() == 32 && dtype.lanes() == 1) {
    int32_t* dims = reinterpret_cast<int32_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else if (dtype.is_int() && dtype.bits() == 64 && dtype.lanes() == 1) {
    int64_t* dims = reinterpret_cast<int64_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else {
    LOG(FATAL) << "invalid shape tensor datatype: " << dtype;
  }

  return shape;
}

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK(exec_) << "The executable is not created yet.";
      std::string func_name = args[0];
      auto git = exec_->global_map.find(func_name);
      ICHECK(git != exec_->global_map.end())
          << "Cannot find function " << func_name << " in the executable";
      auto func = exec_->functions[git->second];
      if (func.params.empty()) {
        *rv = Invoke(func, {});
      } else {
        auto it = inputs_.find(func_name);
        ICHECK(it != inputs_.end()) << "Input has not been set for function " << func_name;
        const std::vector<ObjectRef>& func_args = it->second;
        *rv = Invoke(func, func_args);
      }
    });
  } else if (name == "invoke_stateful") {
    // TODO(tkonolige, jroesch, tqchen): invoke_stateful and get_output are
    // stop-gap measure to allow using vm over a remote connection.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      PackedFunc invoke = GetFunction("invoke", sptr_to_self);
      TVMRetValue rv_;
      invoke.CallPacked(args, &rv_);
    });
  } else if (name == "get_output") {
    return TypedPackedFunc<NDArray(int64_t)>([this](int64_t index) {
      return Downcast<NDArray>(Downcast<ADT>(this->return_register_)[index]);
    });
  } else if (name == "get_num_outputs") {
    return TypedPackedFunc<int64_t(void)>(
        [this]() -> int64_t { return Downcast<ADT>(this->return_register_).size(); });
  } else if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size() % 3, 0);
      std::vector<Device> devices;
      std::vector<AllocatorType> alloc_types;
      for (int i = 0; i < args.size() / 3; ++i) {
        Device dev;
        int device_type = args[i * 3];
        dev.device_type = DLDeviceType(device_type);
        dev.device_id = args[i * 3 + 1];
        int type = args[i * 3 + 2];
        devices.push_back(dev);
        alloc_types.push_back(AllocatorType(type));
      }
      this->Init(devices, alloc_types);
    });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK(exec_) << "The executable is not created yet.";
      std::string func_name = args[0];
      auto gvit = exec_->global_map.find(func_name);
      ICHECK(gvit != exec_->global_map.end()) << "Cannot find function " << func_name;
      auto func_index = gvit->second;
      const auto& vm_func = exec_->functions[func_index];
      const auto& param_names = vm_func.params;
      ICHECK_EQ(args.size() - 1, param_names.size())
          << "The number of provided parameters doesn't match the number of arguments";
      ICHECK_EQ(param_names.size(), vm_func.params_device_type.size())
          << "The number of provided parameters doesn't match the number of assigned devices";
      std::vector<ObjectRef> func_args(param_names.size());
      for (int i = 1; i < args.size(); ++i) {
        Index device_type = vm_func.params_device_type[i - 1];
        Device dev = GetDevice(device_type);

        if (args[i].type_code() == kTVMDLTensorHandle) {
          // Automatically convert input DLTensors to NDArray
          DLTensor* tensor = args[i];
          std::vector<int64_t> shape;
          for (int64_t i = 0; i < tensor->ndim; i++) {
            shape.push_back(tensor->shape[i]);
          }
          NDArray ary = NDArray::Empty(shape, tensor->dtype, dev);
          ary.CopyFrom(tensor);
          func_args[i - 1] = ary;
        } else {
          ObjectRef obj = CopyTo(args[i], dev);
          func_args[i - 1] = obj;
        }
      }
      inputs_.erase(func_name);
      inputs_.emplace(func_name, func_args);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

inline Device VirtualMachine::GetDevice(Index device_type) const {
  ICHECK_GE(devices_.size(), device_type) << "devices_ doesn't contain device:" << device_type;

  auto dev = devices_[device_type];
  ICHECK_EQ(static_cast<Index>(dev.device_type), device_type)
      << "device type " << device_type << " has not been initialized in the device list.";
  return dev;
}

void VirtualMachine::PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index_, arg_count, code_, vm_func.register_file_size);
  frames_.push_back(frame);
}

Index VirtualMachine::PopFrame() {
  ICHECK_GT(frames_.size(), 0);
  const VMFrame& fr = frames_.back();
  func_index_ = fr.func_index;
  code_ = fr.code;
  pc_ = fr.pc;
  auto call_stack_size = frames_.size();
  frames_.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<ObjectRef>& args) {
  DLOG(INFO) << "Invoking global " << func.name << " " << args.size();

  PushFrame(func.params.size(), this->pc_ + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  DLOG(INFO) << "func.params= " << func.params.size();

  code_ = func.instructions.data();
  pc_ = 0;
}

ObjectRef VirtualMachine::Invoke(const VMFunction& func, const std::vector<ObjectRef>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func;

  InvokeGlobal(func, args);
  RunLoop();
  return return_register_;
}

ObjectRef VirtualMachine::Invoke(const std::string& name, const std::vector<ObjectRef>& args) {
  ICHECK(exec_) << "The executable has not been created yet.";
  auto it = exec_->global_map.find(name);
  ICHECK(it != exec_->global_map.end()) << "Cannot find function " << name << " in the executable";
  auto func_index_ = it->second;
  DLOG(INFO) << "Invoke Global " << name << " at index " << func_index_;
  return Invoke(exec_->functions[func_index_], args);
}

void VirtualMachine::InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count,
                                  Index output_size, const std::vector<ObjectRef>& args) {
  size_t arity = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* obj = args[i].as<ADTObj>()) {
      arity += obj->size;
    } else {
      ++arity;
    }
  }

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  int idx = 0;
  bool is_empty_output = false;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* dt_cell = args[i].as<ADTObj>()) {
      for (size_t fi = 0; fi < dt_cell->size; ++fi) {
        auto obj = (*dt_cell)[fi];
        auto nd_array = Downcast<NDArray>(obj);
        setter(idx++, nd_array);
      }
    } else {
      auto nd_array = Downcast<NDArray>(args[i]);
      // We can safely skip CallPacked if there is only one
      // output and it is empty.
      if (i == arg_count - 1 && output_size == 1) {
        for (const auto& dim : nd_array.Shape()) {
          if (!dim) {
            is_empty_output = true;
            break;
          }
        }
      }
      setter(idx++, nd_array);
    }
  }

  if (!is_empty_output) {
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
  }
}

void VirtualMachine::LoadExecutable(const Executable* exec) {
  ICHECK(exec) << "The executable is not created yet.";
  exec_ = exec;

  runtime::Module lib = exec_->GetLib();

  ICHECK(exec->primitive_map.empty() || lib.operator->())
      << "If the executable has declared primitive functions, the"
      << "generated kernel library must non-be null.";

  for (const auto& it : exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (packed_funcs_.size() <= packed_index) {
      packed_funcs_.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, true);
    ICHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    packed_funcs_[packed_index] = pf;
  }
  for (size_t i = 0; i < packed_funcs_.size(); ++i) {
    ICHECK(packed_funcs_[i] != nullptr) << "Packed function " << i << " is not initialized";
  }
}

void VirtualMachine::Init(const std::vector<Device>& devs,
                          const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(devs.size(), alloc_types.size());
  // Cache the device
  for (size_t i = 0; i < devs.size(); i++) {
    auto dev_type = static_cast<size_t>(devs[i].device_type);
    auto alloc = MemoryManager::GetOrCreateAllocator(devs[i], alloc_types[i]);
    if (devices_.size() <= dev_type) {
      devices_.resize(dev_type + 1);
      allocators_.resize(dev_type + 1);
    }
    devices_[dev_type] = devs[i];
    allocators_[dev_type] = alloc;
  }
}

inline void VirtualMachine::WriteRegister(Index r, const ObjectRef& val) {
  frames_.back().register_file[r] = val;
}

inline ObjectRef VirtualMachine::ReadRegister(Index r) const {
  return frames_.back().register_file[r];
}

inline int64_t VirtualMachine::LoadScalarInt(Index r) const {
  int64_t result = 0;
  const auto& obj = ReadRegister(r);
  NDArray array = Downcast<NDArray>(CopyTo(obj, {kDLCPU, 0}));

  switch (array->dtype.bits) {
    case 1: {
      result = reinterpret_cast<bool*>(array->data)[0];
      break;
    }
    case 8: {
      result = reinterpret_cast<int8_t*>(array->data)[0];
      break;
    }
    case 16: {
      result = reinterpret_cast<int16_t*>(array->data)[0];
      break;
    }
    case 32: {
      result = reinterpret_cast<int32_t*>(array->data)[0];
      break;
    }
    case 64: {
      result = reinterpret_cast<int64_t*>(array->data)[0];
      break;
    }
    default:
      LOG(FATAL) << "Unknown scalar int type: " << DLDataType2String(array->dtype);
  }
  return result;
}

void VirtualMachine::RunLoop() {
  ICHECK(this->exec_);
  ICHECK(this->code_);
  pc_ = 0;
  // Index frame_start = frames_.size();
  while (true) {
  main_loop:
    auto const& instr = code_[this->pc_];
    DLOG(INFO) << "Executing(" << pc_ << "): " << instr;

    switch (instr.op) {
      case Opcode::CallPacked: {
        std::vector<ObjectRef> args;
        for (Index i = 0; i < instr.num_args; ++i) {
          args.push_back(ReadRegister(instr.args[i]));
        }
        InvokeGlobal(exec_->functions[instr.packed_index], args);
        frames_.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      default:
        LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
    }
  }
}

runtime::Module CreateVirtualMachine(const Executable* exec) {
  auto vm = make_object<VirtualMachine>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._NewVirtualMachine").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec);
});

TVM_REGISTER_GLOBAL("runtime._NewVirtualMachineTest").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec);
});

}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
