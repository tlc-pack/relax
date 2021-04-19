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
 * \file tvm/runtime/vm/executable.cc
 * \brief The implementation of a virtual machine executable APIs.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "./executable.h"
#include "./vm.h"
#include "./serialize_utils.h"
#include "../runtime/file_utils.h"
#include "../runtime/library_module.h"

namespace tvm {
namespace runtime {
namespace new_vm {

#define STREAM_CHECK(val, section)                                          \
  ICHECK(val) << "Invalid VM file format in the " << section << " section." \
              << "\n";

// Helper to serialize a vm instruction.
VMInstructionSerializer SerializeInstruction(const Instruction& instr);
// Helper to deserialize a serialized vm instruction.
Instruction DeserializeInstruction(const VMInstructionSerializer& instr);

PackedFunc Executable::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_lib") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetLib(); });
  } else if (name == "get_bytecode") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetBytecode(); });
  } else if (name == "get_stats") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->Stats(); });
  } else if (name == "save") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->Save(); });
  } else if (name == "get_function_arity") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      *rv = this->GetFunctionArity(func_name);
    });
  } else if (name == "get_function_param_name") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      int index = args[1];
      *rv = this->GetFunctionParameterName(func_name, index);
    });
  } else if (name == "vm_load_executable") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      auto vm = make_object<VirtualMachine>();
      vm->LoadExecutable(this);
      *rv = Module(vm);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc(nullptr);
  }
}

int Executable::GetFunctionArity(std::string func_name) const {
  auto it = global_map.find(func_name);
  if (it == global_map.end()) {
    LOG(ERROR) << "Cannot find function " << func_name << " in executable";
    return -1;
  }
  const auto& func = functions[it->second];
  return func.params.size();
}

std::string Executable::GetFunctionParameterName(std::string func_name, uint32_t index) const {
  auto it = global_map.find(func_name);
  if (it == global_map.end()) {
    LOG(ERROR) << "Cannot find function " << func_name << " in executable";
    return "";
  }
  const auto& func = functions[it->second];
  if (index > func.params.size()) {
    LOG(ERROR) << "Invalid parameter index";
    return "";
  }
  return func.params[index];
}

std::string Executable::GetBytecode() const {
  std::ostringstream oss;

  for (size_t i = 0; i < functions.size(); ++i) {
    const auto& func = functions[i];
    // Print the header of the function format.
    oss << "VM Function[" << i << "]: " << func.name << "(";
    for (const auto& param : func.params) {
      oss << param << ", ";
    }
    oss.seekp(-2, std::ios_base::end);
    oss << ")" << std::endl;
    oss << "# reg file size = " << func.register_file_size << std::endl;
    oss << "# instruction count = " << func.instructions.size() << std::endl;

    // Print the instructions of a `VMFunction`.
    // The part after ";" is the instruction in text format.
    oss << "opcode, fields # inst(text):" << std::endl;
    for (size_t idx = 0; idx < func.instructions.size(); ++idx) {
      const auto& instr = func.instructions[idx];
      const auto& serialized_instr = SerializeInstruction(instr);
      oss << std::setw(2) << idx << ": " << serialized_instr.opcode << " ";
      for (auto it : serialized_instr.fields) {
        oss << it << " ";
      }
      oss << "  # " << instr;
      if (oss.str().back() != '\n') oss << std::endl;
    }
    oss << std::endl;
  }

  return oss.str();
}

std::string Executable::Stats() const {
  std::ostringstream oss;
  oss << "VM executable statistics:" << std::endl;

  // Get the number of constants and the shape of each of them.
  oss << "  Constant shapes (# " << constants.size() << "): [";
  for (const auto& it : constants) {
    const auto constant = Downcast<NDArray>(it);
    const auto& shape = constant.Shape();

    // Scalar
    if (shape.empty()) {
      oss << "scalar, ";
      continue;
    }

    oss << "[";
    for (auto s : shape) {
      oss << s << ", ";
    }
    oss.seekp(-2, oss.cur);
    oss << "], " << std::endl;
  }
  if (!constants.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of globals and the name of each of them.
  oss << "  Globals (#" << global_map.size() << "): [";
  for (const auto& it : global_map) {
    oss << "(\"" << it.first << "\", " << it.second << ")"
        << ", ";
  }
  if (!global_map.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of primitive ops and the name of each of them.
  oss << "  Primitive ops (#" << primitive_map.size() << "): [";
  std::vector<std::string> prim_ops;
  for (const auto& it : primitive_map) {
    auto packed_index = static_cast<size_t>(it.second);
    if (prim_ops.size() <= packed_index) {
      prim_ops.resize(packed_index + 1);
    }
    prim_ops[packed_index] = it.first;
  }
  for (const auto& it : prim_ops) {
    oss << it << ", ";
  }
  if (!prim_ops.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  return oss.str();
}

void SaveHeader(dmlc::Stream* strm) {
  uint64_t header = kTVMVMBytecodeMagic;
  strm->Write(header);
  std::string version = TVM_VERSION;
  strm->Write(version);
}

TVMByteArray Executable::Save() {
  // Initialize the stream object.
  code_.clear();
  dmlc::MemoryStringStream strm(&code_);

  // Save header
  SaveHeader(&strm);

  // Global section.
  SaveGlobalSection(&strm);

  // Constant section.
  SaveConstantSection(&strm);

  // Primitive names.
  SavePrimitiveOpNames(&strm);

  // Code section.
  SaveCodeSection(&strm);

  TVMByteArray arr;
  arr.data = code_.c_str();
  arr.size = code_.length();
  return arr;
}

void Executable::SaveGlobalSection(dmlc::Stream* strm) {
  std::vector<std::pair<std::string, Index> > globals(this->global_map.begin(),
                                                      this->global_map.end());
  auto comp = [](const std::pair<std::string, Index>& a, const std::pair<std::string, Index>& b) {
    return a.second < b.second;
  };
  std::sort(globals.begin(), globals.end(), comp);

  std::vector<std::string> glbs;
  for (const auto& it : globals) {
    glbs.push_back(it.first);
  }
  strm->Write(glbs);
}

void Executable::SaveConstantSection(dmlc::Stream* strm) {
  std::vector<DLTensor*> arrays;
  for (const auto& obj : this->constants) {
    const auto cell = Downcast<runtime::NDArray>(obj);
    arrays.push_back(const_cast<DLTensor*>(cell.operator->()));
  }
  strm->Write(static_cast<uint64_t>(this->constants.size()));
  for (const auto& it : arrays) {
    runtime::SaveDLTensor(strm, it);
  }

  // Save the const to device mapping.
  strm->Write(this->const_device_type);
}

void Executable::SavePrimitiveOpNames(dmlc::Stream* strm) {
  std::vector<std::string> primitive_names;
  for (const auto& it : this->primitive_map) {
    auto packed_index = static_cast<size_t>(it.second);
    if (primitive_names.size() <= packed_index) {
      primitive_names.resize(packed_index + 1);
    }
    primitive_names[packed_index] = it.first;
  }
  strm->Write(primitive_names);
}

// Serialize a virtual machine instruction. It creates a list that contains the
// hash, opcode, and all fields of an instruction.
//
// For example, the function signature used to create an `AllocTensor`
// instruction is:
//   Instruction AllocTensor(std::vector<Index> shape, DLDataType dtype, RegName dst)
//
// The serialized form will be:
//   `hash 5 dtype.code dtype.bits dtype.lanes ndim dst_register val1 val2 ... valn`
//
// where hash is the hash of serialized instruction that is computed internally
// by the `VMInstructionExecutable`. It is used for sanity check before decoding.
// 5 shows opcode of `AllocTensor`, `(dtype.code dtype.bits dtype.lanes)`
// represents a `DLDataType`, `ndim` is the number of dimensions, `dst_register`
// is the destination register, and the rest of it together indicates the shape
// of the tensor to be allocated.
VMInstructionSerializer SerializeInstruction(const Instruction& instr) {
  std::vector<Index> fields;
  // Save the opcode.
  DLOG(INFO) << "Serializing: " << instr << std::endl;
  switch (instr.op) {
    case Opcode::CallPacked: {
      fields.assign({instr.packed_index, instr.num_args});
      fields.insert(fields.end(), instr.args, instr.args + instr.num_args);
      break;
    }
    default:
      LOG(FATAL) << "Invalid opcode" << static_cast<int>(instr.op);
      break;
  }

  return VMInstructionSerializer(static_cast<Index>(instr.op), fields);
}

void Executable::SaveCodeSection(dmlc::Stream* strm) {
  // Save the number of functions.
  strm->Write(static_cast<uint64_t>(this->functions.size()));
  for (const auto& func : this->functions) {
    // Save the function info.
    VMFunctionSerializer func_format(func.name, func.register_file_size, func.instructions.size(),
                                     func.params, func.params_device_type);
    func_format.Save(strm);

    // Serialize each instruction.
    for (const auto& instr : func.instructions) {
      const auto& serialized_instr = SerializeInstruction(instr);
      serialized_instr.Save(strm);
    }
  }
}

void LoadHeader(dmlc::Stream* strm) {
  // Check header.
  uint64_t header;
  STREAM_CHECK(strm->Read(&header), "header");
  STREAM_CHECK(header == kTVMVMBytecodeMagic, "header");

  // Check version.
  std::string version;
  STREAM_CHECK(strm->Read(&version), "version");
  STREAM_CHECK(version == TVM_VERSION, "version");
}

runtime::Module Executable::GetLib() const {
  ICHECK_LE(this->imports_.size(), 1)
      << "The kernel library must be imported as the only module in an Executable";

  if (this->imports().size() == 0) {
    return Module(nullptr);
  } else {
    return this->imports_[0];
  }
}

void Executable::SetLib(const runtime::Module& lib) {
  ICHECK(lib.defined()) << "the provided library can not be null";

  ICHECK_EQ(this->imports_.size(), 0)
      << "A VMExecutable should never have more than one import inside an the executable, \n"
      << "the first import should *always* be the library containing"
      << "the platform specific kernel code";

  this->Import(lib);
}

runtime::Module Executable::Load(const std::string& code, const runtime::Module lib) {
  auto exec = make_object<Executable>();

  // Support null-initialization of lib, to enable initialization during
  // deserialization before we have we have deserialized the imports.
  if (lib.defined()) {
    exec->SetLib(lib);
  }

  exec->code_ = code;
  dmlc::MemoryStringStream strm(&exec->code_);

  // Load header.
  LoadHeader(&strm);

  // Global section.
  exec->LoadGlobalSection(&strm);

  // Constant section.
  exec->LoadConstantSection(&strm);

  // Primitive names that will be invoked by `InvokePacked` instructions.
  exec->LoadPrimitiveOpNames(&strm);

  // Code section.
  exec->LoadCodeSection(&strm);

  return runtime::Module(exec);
}

void Executable::LoadGlobalSection(dmlc::Stream* strm) {
  std::vector<std::string> globals;
  STREAM_CHECK(strm->Read(&globals), "global");
  for (size_t i = 0; i < globals.size(); i++) {
    this->global_map.insert({globals[i], i});
  }
}

void Executable::LoadConstantSection(dmlc::Stream* strm) {
  uint64_t sz;
  // Load the number of constants.
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "constant");

  size_t size = static_cast<size_t>(sz);
  // Load each of the constants.
  for (size_t i = 0; i < size; i++) {
    runtime::NDArray constant;
    STREAM_CHECK(constant.Load(strm), "constant");
    this->constants.push_back(constant);
  }

  // Load the const to device mapping.
  std::vector<Index> const_device_type;
  STREAM_CHECK(strm->Read(&const_device_type), "constant");
  ICHECK_EQ(size, const_device_type.size());
  this->const_device_type = const_device_type;
}

void Executable::LoadPrimitiveOpNames(dmlc::Stream* strm) {
  std::vector<std::string> primitive_names;
  STREAM_CHECK(strm->Read(&primitive_names), "primitive name");
  for (size_t i = 0; i < primitive_names.size(); i++) {
    this->primitive_map.insert({primitive_names[i], i});
  }
}

// Extract the `cnt` number of fields started at `start` from the list
// `instr_fields`.
inline std::vector<Index> ExtractFields(const std::vector<Index>& instr_fields, Index start,
                                        Index cnt) {
  ICHECK_LE(static_cast<size_t>(start + cnt), instr_fields.size());
  std::vector<Index> ret;
  for (auto i = start; i < start + cnt; i++) {
    ret.push_back(instr_fields[i]);
  }
  return ret;
}

Instruction DeserializeInstruction(const VMInstructionSerializer& instr) {
  Opcode opcode = static_cast<Opcode>(instr.opcode);
  switch (opcode) {
    case Opcode::CallPacked: {
      DCHECK_GE(instr.fields.size(), 2U);
      DCHECK_EQ(instr.fields.size(), 2U + static_cast<size_t>(instr.fields[1]));
      Index packed_index = instr.fields[0];
      Index num_args = instr.fields[1];
      std::vector<RegName> args = ExtractFields(instr.fields, 2, num_args);
      return Instruction::CallPacked(packed_index, num_args, args);
    }
    default:
      LOG(FATAL) << "Invalid opcode" << instr.opcode;
      return Instruction();
  }
}

void Executable::LoadCodeSection(dmlc::Stream* strm) {
  // Load the number of functions.
  uint64_t sz;
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "code");

  size_t num_funcs = static_cast<size_t>(sz);
  this->functions.resize(num_funcs);
  for (size_t i = 0; i < num_funcs; i++) {
    // Load the function info.
    VMFunctionSerializer loaded_func;
    STREAM_CHECK(loaded_func.Load(strm), "code/function");

    // Load the instructions.
    std::vector<Instruction> instructions;
    for (size_t j = 0; j < loaded_func.num_instructions; j++) {
      VMInstructionSerializer instr;
      std::vector<Index> instr_fields;
      STREAM_CHECK(instr.Load(strm), "code/instruction");
      instructions.push_back(DeserializeInstruction(instr));
    }

    // Create the VM function.
    VMFunction vm_func = VMFunction(loaded_func.name, loaded_func.params, instructions,
                                    loaded_func.register_file_size, loaded_func.params_device_type);
    auto it = this->global_map.find(loaded_func.name);
    ICHECK(it != this->global_map.end());
    ICHECK_LE(it->second, this->global_map.size());
    this->functions[it->second] = vm_func;
  }
}

void Executable::SaveToBinary(dmlc::Stream* stream) {
  auto code_bytes = this->Save();
  std::string code(code_bytes.data, code_bytes.size);
  stream->Write(code);

  ICHECK(this->imports()[0].defined()) << "the library must be imported before serialization";
}

Module ExecutableLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string code;
  stream->Read(&code);
  auto exec = Executable::Load(code, Module());
  return exec;
}

void Executable::SaveToFile(const std::string& path, const std::string& format) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  dmlc::SeekStream* strm = &writer;
  SaveToBinary(strm);
  SaveBinaryToFile(path, data);
}

TVM_REGISTER_GLOBAL("runtime.module.new_loadbinary_VMExecutable").
set_body_typed(ExecutableLoadBinary);

// Load module from module.
Module ExecutableLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  LoadBinaryFromFile(file_name, &data);
  dmlc::MemoryStringStream reader(&data);
  dmlc::Stream* strm = &reader;
  auto exec = ExecutableLoadBinary(reinterpret_cast<void*>(strm));
  return exec;
}

TVM_REGISTER_GLOBAL("runtime.module.new_loadfile_VMExecutable").set_body_typed(ExecutableLoadFile);

TVM_REGISTER_GLOBAL("runtime.NewGetNumOfGlobals").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  *rv = static_cast<int>(exec->global_map.size());
});

TVM_REGISTER_GLOBAL("runtime.NewGetGlobalFields").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  int idx = args[1];
  std::vector<std::pair<std::string, Index> > globals(exec->global_map.begin(),
                                                      exec->global_map.end());
  auto comp = [](const std::pair<std::string, Index>& a, const std::pair<std::string, Index>& b) {
    return a.second < b.second;
  };
  std::sort(globals.begin(), globals.end(), comp);
  ICHECK_LT(idx, globals.size());
  *rv = globals[idx].first;
});

TVM_REGISTER_GLOBAL("runtime.NewGetNumOfPrimitives").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  *rv = static_cast<int>(exec->primitive_map.size());
});

TVM_REGISTER_GLOBAL("runtime.NewGetPrimitiveFields").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  int idx = args[1];
  ICHECK_GE(idx, 0);
  ICHECK_LT(idx, exec->primitive_map.size());

  for (const auto& it : exec->primitive_map) {
    if (idx == static_cast<int>(it.second)) {
      *rv = it.first;
      break;
    }
  }
});

TVM_REGISTER_GLOBAL("runtime.NewLoadExecutable")
    .set_body_typed([](std::string code, runtime::Module lib) {
      return Executable::Load(code, lib);
    });

}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
