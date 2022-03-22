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
 * \file src/runtime/relax_vm/executable.cc
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <functional>
#include <sstream>

#include "../file_utils.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

/*! \brief The magic number for the serialized VM bytecode file  */
constexpr uint64_t kTVMVMBytecodeMagic = 0xD225DE2F4214151D;

/*! \brief Possible types in the constant pool */
enum ConstantType : int {
  kNDArray = 0,
  kDLDataType = 1,
  kShapeTuple = 2,
  kString = 3,
};

#define STREAM_CHECK(val, section)                                          \
  ICHECK(val) << "Invalid VM file format in the " << section << " section." \
              << "\n";

PackedFunc Executable::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "stats") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->Stats(); });
  } else if (name == "save_to_file") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1);
      std::string path = args[0];
      this->SaveToFile(path, "");
    });
  } else if (name == "as_text") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->AsText(); });
  } else if (name == "as_python") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->AsPython(); });
  } else if (name == "vm_load_executable") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ObjectPtr<VirtualMachine> vm = make_object<VirtualMachine>();
      ICHECK(sptr_to_self.get() == this);
      vm->LoadExecutable(GetObjectPtr<Executable>(this));
      *rv = Module(vm);
    });
  }
  return nullptr;
}

std::string Executable::Stats() const {
  std::ostringstream oss;
  oss << "Relax VM executable statistics:" << std::endl;

  // Get the number of constants.
  // If the constant is an NDArray, get the shape of each of them.
  // If the constant is an DLDataType, get the data type of each of them.
  oss << "  Constant pool (# " << constants.size() << "): [";
  for (const auto& it : constants) {
    if (it.IsObjectRef<runtime::NDArray>()) {
      const auto ndarray = it.operator tvm::runtime::NDArray();
      const auto& shape = ndarray.Shape();
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
      oss << "], ";
    } else if (it.IsObjectRef<ShapeTuple>()) {
      ShapeTuple shape = it.operator ShapeTuple();
      oss << "shapetuple[";
      for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape.at(i) << ", ";
      }
      oss.seekp(-2, oss.cur);
      oss << "], ";
    } else if (it.IsObjectRef<String>()) {
      std::string f = it.AsObjectRef<tvm::runtime::String>().operator std::string();
      oss << "\"";
      oss << f;
      oss << "\", ";
    } else {
      try {
        DLDataType dtype = it.operator DLDataType();
        oss << dtype;
        oss << ", ";
      } catch (std::exception& exc) {
        LOG(FATAL) << "Constant pool can only contain NDArray and DLDataType, but got "
                   << ArgTypeCode2Str(it.type_code());
      }
    }
  }
  if (!constants.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of globals and the name of each of them.
  oss << "  Globals (#" << global_funcs.size() << "): [";
  for (const auto& it : global_funcs) {
    oss << it.name << ", ";
  }
  if (!global_map.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of packed funcs and the name of each of them.
  oss << "  Packed functions (#" << func_names.size() << "): [";
  for (const auto& it : func_names) {
    oss << it << ", ";
  }
  if (!func_names.empty()) {
    oss.seekp(-2, oss.cur);
  }
  oss << "]" << std::endl;

  return oss.str();
}

void Executable::SetInstructionData(Index i, Index j, ExecWord val) {
  Index instr_idx = instr_offset[i];
  instr_data[instr_idx + j] = val;
}

Instruction Executable::GetInstruction(Index i) const {
  size_t offset = instr_offset[i];
  Opcode op = static_cast<Opcode>(instr_data[offset]);
  switch (op) {
    case Opcode::Call: {
      RegName dst = instr_data[offset + 1];
      Index func_idx = instr_data[offset + 2];
      Index num_args = instr_data[offset + 3];
      ExecWord* args = const_cast<ExecWord*>(&instr_data[offset + 4]);
      return Instruction::Call(func_idx, num_args, reinterpret_cast<Instruction::Arg*>(args), dst);
    }
    case Opcode::Ret: {
      RegName result = instr_data[offset + 1];
      return Instruction::Ret(result);
    }
    case Opcode::Goto: {
      Index pc_offset = instr_data[offset + 1];
      return Instruction::Goto(pc_offset);
    }
    case Opcode::If: {
      RegName cond = instr_data[offset + 1];
      Index false_offset = instr_data[offset + 2];
      return Instruction::If(cond, false_offset);
    }
    default:
      LOG(FATAL) << "should never hit this case: " << static_cast<int>(op);
      break;
  }
  return Instruction();
}

void SaveHeader(dmlc::Stream* strm) {
  uint64_t header = kTVMVMBytecodeMagic;
  strm->Write(header);
  std::string version = TVM_VERSION;
  strm->Write(version);
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

void Executable::SaveToBinary(dmlc::Stream* stream) {
  std::string code;
  // Initialize the stream object.
  dmlc::MemoryStringStream strm(&code);

  // Save header
  SaveHeader(&strm);

  // Global section.
  SaveGlobalSection(&strm);

  // Constant section.
  SaveConstantSection(&strm);

  // Packedfunc names section.
  SavePackedFuncNames(&strm);

  // Code section.
  SaveCodeSection(&strm);

  stream->Write(code);
}

void Executable::SaveToFile(const std::string& file_name, const std::string& format) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  dmlc::SeekStream* strm = &writer;
  Executable::SaveToBinary(strm);
  runtime::SaveBinaryToFile(file_name, data);
}

Module Executable::LoadFromBinary(void* stream) {
  std::string code;
  static_cast<dmlc::Stream*>(stream)->Read(&code);
  dmlc::MemoryStringStream strm(&code);

  ObjectPtr<Executable> exec = make_object<Executable>();

  // Load header.
  LoadHeader(&strm);

  // Global section.
  exec->LoadGlobalSection(&strm);

  // Constant section.
  exec->LoadConstantSection(&strm);

  // Packedfunc names section.
  exec->LoadPackedFuncNames(&strm);

  // Code section.
  exec->LoadCodeSection(&strm);

  return Module(exec);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_relax.Executable")
    .set_body_typed(Executable::LoadFromBinary);

Module Executable::LoadFromFile(const std::string& file_name) {
  std::string data;
  runtime::LoadBinaryFromFile(file_name, &data);
  dmlc::MemoryStringStream reader(&data);
  dmlc::Stream* strm = &reader;
  return Executable::LoadFromBinary(reinterpret_cast<void*>(strm));
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_relax.Executable")
    .set_body_typed(Executable::LoadFromFile);

void SerializeVMFunc(const VMFunction& func, dmlc::Stream* strm) {
  strm->Write(func.name);
  strm->Write(func.start_instr);
  strm->Write(func.num_args);
  strm->Write(func.register_file_size);
}

VMFunction DeserializeVMFunc(dmlc::Stream* strm) {
  VMFunction func;
  STREAM_CHECK(strm->Read(&func.name), "vmfunc name");
  STREAM_CHECK(strm->Read(&func.start_instr), "vmfunc start_instr");
  STREAM_CHECK(strm->Read(&func.num_args), "vmfunc num_args");
  STREAM_CHECK(strm->Read(&func.register_file_size), "vmfunc register_file_size");
  return func;
}

void Executable::SaveGlobalSection(dmlc::Stream* strm) {
  strm->Write(static_cast<uint64_t>(this->global_funcs.size()));
  for (const auto& func : this->global_funcs) {
    SerializeVMFunc(func, strm);
  }
}

void Executable::SaveConstantSection(dmlc::Stream* strm) {
  strm->Write(static_cast<uint64_t>(this->constants.size()));
  for (const auto& it : this->constants) {
    if (it.IsObjectRef<runtime::NDArray>()) {
      strm->Write(ConstantType::kNDArray);
      runtime::SaveDLTensor(strm, it.operator DLTensor*());
    } else if (it.IsObjectRef<ShapeTuple>()) {
      ShapeTuple shape = it.operator ShapeTuple();
      strm->Write(ConstantType::kShapeTuple);
      strm->Write(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        strm->Write(shape.at(i));
      }
    } else if (it.IsObjectRef<String>()) {
      String str = it.operator String();
      strm->Write(ConstantType::kString);
      strm->Write(str.size());
      for (size_t i = 0; i < str.size(); ++i) {
        strm->Write(str.at(i));
      }
    } else {
      try {
        strm->Write(ConstantType::kDLDataType);
        strm->Write(it.operator DLDataType());
      } catch (std::exception& exc) {
        LOG(FATAL) << "Constant pool can only contain NDArray and DLDataType, but got "
                   << ArgTypeCode2Str(it.type_code());
      }
    }
  }
}

void Executable::SavePackedFuncNames(dmlc::Stream* strm) { strm->Write(func_names); }

void Executable::SaveCodeSection(dmlc::Stream* strm) {
  strm->Write(instr_offset);
  strm->Write(instr_data);
}

void Executable::LoadGlobalSection(dmlc::Stream* strm) {
  uint64_t sz;
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "constant");
  size_t size = static_cast<size_t>(sz);
  for (size_t i = 0; i < size; i++) {
    VMFunction func = DeserializeVMFunc(strm);
    this->global_funcs.push_back(func);
  }
  for (size_t i = 0; i < global_funcs.size(); ++i) {
    this->global_map[global_funcs[i].name] = i;
  }
}

void Executable::LoadConstantSection(dmlc::Stream* strm) {
  uint64_t sz;
  // Load the number of constants.
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "constant");

  size_t size = static_cast<size_t>(sz);
  runtime::NDArray ndarray;
  DLDataType dtype;
  // Load each of the constants.
  for (size_t i = 0; i < size; i++) {
    int constant_type;
    STREAM_CHECK(strm->Read(&constant_type, sizeof(constant_type)), "constant");
    if (constant_type == ConstantType::kNDArray) {
      ndarray.Load(strm);
      TVMRetValue cell;
      cell = ndarray;
      this->constants.push_back(cell);
    } else if (constant_type == ConstantType::kShapeTuple) {
      size_t size;
      strm->Read(&size);
      std::vector<ShapeTuple::index_type> data(size);
      for (size_t i = 0; i < size; ++i) {
        strm->Read(&(data[i]));
      }
      TVMRetValue cell;
      cell = ShapeTuple(data);
      this->constants.push_back(cell);
    } else if (constant_type == ConstantType::kDLDataType) {
      strm->Read(&dtype);
      TVMRetValue cell;
      cell = dtype;
      this->constants.push_back(cell);
    } else if (constant_type == ConstantType::kString) {
      size_t size;
      strm->Read(&size);
      std::vector<char> data(size);
      for (size_t i = 0; i < size; ++i) {
        strm->Read(&(data[i]));
      }
      TVMRetValue cell;
      cell = String(std::string(data.begin(), data.end()));
      this->constants.push_back(cell);
    } else {
      LOG(FATAL) << "Constant pool can only contain NDArray and DLDataType, but got "
                 << ArgTypeCode2Str(constant_type) << " when loading the VM constant pool.";
    }
  }
}

void Executable::LoadPackedFuncNames(dmlc::Stream* strm) {
  STREAM_CHECK(strm->Read(&(this->func_names)), "packed func names");
  for (size_t i = 0; i < func_names.size(); ++i) {
    this->func2idx[func_names[i]] = i;
  }
}

void Executable::LoadCodeSection(dmlc::Stream* strm) {
  STREAM_CHECK(strm->Read(&(this->instr_offset)), "instr offset");
  STREAM_CHECK(strm->Read(&(this->instr_data)), "instr data");
}

template <typename T>
std::string StrJoin(T* items, int offset, int cnt, std::string delim = ", ",
                    std::function<std::string(T)> repr = std::to_string) {
  if (cnt == 0) {
    return "";
  }
  std::ostringstream oss;
  oss << repr(items[offset]);
  for (int i = 1; i < cnt; ++i) {
    oss << delim << repr(items[offset + i]);
  }
  return oss.str();
}

std::string RegNameToStr(RegName reg) {
  if (reg == Instruction::kVoidArg) {
    return "void";
  }
  if (reg == Instruction::kVMStateRegister) {
    return "%state";
  }
  return "%" + std::to_string(reg);
}

std::string InstrArgToStr(Instruction::Arg arg) {
  // only for argument
  switch (arg.kind()) {
    case Instruction::kRegister:
      return RegNameToStr(arg.value());
    case Instruction::kImmediate:
      return "i" + std::to_string(arg.value());
    case Instruction::kConstIdx:
      return "c[" + std::to_string(arg.value()) + "]";
    default:
      LOG(FATAL) << "Wrong instruction kind: " << arg.kind();
      return "";
  }
}

std::string InstrArgToPyStr(Instruction::Arg arg) {
  switch (arg.kind()) {
    case Instruction::kRegister:
      if (arg.value() == Instruction::kVMStateRegister) {
        return "ib.r(state)";
      }
      return "ib.r(" + std::to_string(arg.value()) + ")";
    case Instruction::kImmediate:
      return "ib.imm(" + std::to_string(arg.value()) + ")";
    case Instruction::kConstIdx:
      return "ib.c(" + std::to_string(arg.value()) + ")";
    default:
      LOG(FATAL) << "Wrong instruction kind: " << arg.kind();
      return "";
  }
}

String Executable::AsText() const {
  // print the text format
  std::ostringstream os;
  for (size_t fidx = 0; fidx < this->global_funcs.size(); ++fidx) {
    const VMFunction& gfunc = this->global_funcs[fidx];
    os << "@" << gfunc.name << ":\n";
    size_t start_instr = gfunc.start_instr;
    size_t end_instr = this->instr_offset.size();
    if ((fidx + 1) < global_funcs.size()) {
      end_instr = global_funcs[fidx + 1].start_instr;
    }
    for (size_t idx = start_instr; idx < end_instr; ++idx) {
      os << "  ";
      Instruction instr = this->GetInstruction(idx);
      switch (instr.op) {
        case Opcode::Call: {
          os << std::setw(6) << std::left << "call" << std::setw(16) << std::left
             << this->func_names[instr.func_idx] << " in: " << std::setw(12) << std::left
             << StrJoin<Instruction::Arg>(instr.args, 0, instr.num_args, ", ", InstrArgToStr)
             << " dst: " << RegNameToStr(instr.dst) << "\n";
          break;
        }
        case Opcode::Ret: {
          os << std::setw(6) << std::left << "ret"
             << "ret " << RegNameToStr(instr.result) << "\n";
          break;
        }
        case Opcode::Goto: {
          os << std::setw(6) << std::left << "goto"
             << "goto " << instr.pc_offset << "\n";
          break;
        }
        case Opcode::If: {
          os << std::setw(6) << std::left << "If" << RegNameToStr(instr.cond) << ", "
             << instr.false_offset << "\n";
          break;
        }
        default:
          LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
          break;
      }
    }
    os << "\n";
  }
  return String(os.str());
}

String Executable::AsPython() const {
  // print the python format
  std::ostringstream os;
  os << "ib = rx.Builder()\n";
  for (size_t fidx = 0; fidx < this->global_funcs.size(); ++fidx) {
    const VMFunction& gfunc = this->global_funcs[fidx];
    os << "with ib.function(\"" << gfunc.name << "\", num_inputs=" << gfunc.num_args << "):\n";
    size_t start_instr = gfunc.start_instr;
    size_t end_instr = this->instr_offset.size();
    if ((fidx + 1) < global_funcs.size()) {
      end_instr = global_funcs[fidx + 1].start_instr;
    }
    for (size_t idx = start_instr; idx < end_instr; ++idx) {
      Instruction instr = this->GetInstruction(idx);
      switch (instr.op) {
        case Opcode::Call: {
          os << "    ib.emit_call(\"" << this->func_names[instr.func_idx] << "\", args=["
             << StrJoin<Instruction::Arg>(instr.args, 0, instr.num_args, ", ", InstrArgToPyStr)
             << "]";
          if (instr.dst != Instruction::kVoidArg) os << ", dst=ib.r(" << instr.dst << ")";
          os << ")\n";
          break;
        }
        case Opcode::Ret: {
          os << "    ib.emit_ret(ib.r(" << instr.result << "))\n";
          break;
        }
        case Opcode::Goto: {
          os << "    ib.emit_goto(" << instr.pc_offset << ")\n";
          break;
        }
        case Opcode::If: {
          os << "    ib.emit_if(ib.r(" << instr.cond << "), " << instr.false_offset << ")\n";
          break;
        }
        default:
          LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
          break;
      }
    }
  }
  return String(os.str());
}

TVM_REGISTER_GLOBAL("relax.ExecutableLoadFromFile").set_body_typed(Executable::LoadFromFile);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
