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
 * \file src/relax/vm/executable.cc
 * \brief
 */

#include <dmlc/memory_io.h>
#include <tvm/relax/vm/executable.h>
#include <tvm/runtime/logging.h>

#include <functional>
#include <sstream>

#include "../../runtime/file_utils.h"

namespace tvm {
namespace relax {
namespace vm {

/*! \brief The magic number for the serialized VM bytecode file  */
constexpr uint64_t kTVMVMBytecodeMagic = 0xD225DE2F4214151D;

#define STREAM_CHECK(val, section)                                          \
  ICHECK(val) << "Invalid VM file format in the " << section << " section." \
              << "\n";

TVM_REGISTER_NODE_TYPE(ExecutableNode);

std::string ExecutableNode::Stats() const {
  std::ostringstream oss;
  oss << "Relax VM executable statistics:" << std::endl;

  // Get the number of constants and the shape of each of them.
  oss << "  Constant shapes (# " << constants.size() << "): [";
  for (const auto& it : constants) {
    const auto constant = Downcast<runtime::NDArray>(it);
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

Instruction ExecutableNode::GetInstruction(Index i) const {
  size_t offset = instr_offset[i];
  Opcode op = static_cast<Opcode>(instr_data[offset]);
  switch (op) {
    case Opcode::Call: {
      RegName dst = instr_data[offset + 1];
      Index func_idx = instr_data[offset + 2];
      Index num_args = instr_data[offset + 3];
      ExecWord* args = const_cast<ExecWord*>(&instr_data[offset + 4]);
      return Instruction::Call(func_idx, num_args, reinterpret_cast<InstrArg*>(args), dst);
    }
    case Opcode::Ret: {
      RegName result = instr_data[offset + 1];
      return Instruction::Ret(result);
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

TVMByteArray ExecutableNode::Save() {
  // Initialize the stream object.
  code_.clear();
  dmlc::MemoryStringStream strm(&code_);

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

  TVMByteArray arr;
  arr.data = code_.c_str();
  arr.size = code_.length();
  return arr;
}

Executable ExecutableNode::Load(const std::string& code) {
  auto exec = make_object<ExecutableNode>();

  exec->code_ = code;
  dmlc::MemoryStringStream strm(&exec->code_);

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

  return Executable(exec);
}

void ExecutableNode::SaveToBinary(dmlc::Stream* stream) {
  auto code_bytes = this->Save();
  std::string code(code_bytes.data, code_bytes.size);
  stream->Write(code);
}

void ExecutableNode::SaveToFile(const std::string& path) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  dmlc::SeekStream* strm = &writer;
  ExecutableNode::SaveToBinary(strm);
  runtime::SaveBinaryToFile(path, data);
}

Executable ExecutableNode::LoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string code;
  stream->Read(&code);
  auto exec = ExecutableNode::Load(code);
  return exec;
}

Executable ExecutableNode::LoadFromFile(const std::string& file_name) {
  std::string data;
  runtime::LoadBinaryFromFile(file_name, &data);
  dmlc::MemoryStringStream reader(&data);
  dmlc::Stream* strm = &reader;
  auto exec = ExecutableNode::LoadFromBinary(reinterpret_cast<void*>(strm));
  return exec;
}

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

void ExecutableNode::SaveGlobalSection(dmlc::Stream* strm) {
  strm->Write(static_cast<uint64_t>(this->global_funcs.size()));
  for (const auto& func : this->global_funcs) {
    SerializeVMFunc(func, strm);
  }
}

void ExecutableNode::SaveConstantSection(dmlc::Stream* strm) {
  std::vector<DLTensor*> arrays;
  for (const auto& obj : this->constants) {
    const auto cell = Downcast<runtime::NDArray>(obj);
    arrays.push_back(const_cast<DLTensor*>(cell.operator->()));
  }
  strm->Write(static_cast<uint64_t>(this->constants.size()));
  for (const auto& it : arrays) {
    runtime::SaveDLTensor(strm, it);
  }
}

void ExecutableNode::SavePackedFuncNames(dmlc::Stream* strm) {
  strm->Write(func_names);
}

void ExecutableNode::SaveCodeSection(dmlc::Stream* strm) {
  strm->Write(instr_offset);
  strm->Write(instr_data);
}

void ExecutableNode::LoadGlobalSection(dmlc::Stream* strm) {
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

void ExecutableNode::LoadConstantSection(dmlc::Stream* strm) {
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
}

void ExecutableNode::LoadPackedFuncNames(dmlc::Stream* strm) {
  STREAM_CHECK(strm->Read(&(this->func_names)), "packed func names");
  for (size_t i = 0; i < func_names.size(); ++i) {
    this->func2idx[func_names[i]] = i;
  }
}

void ExecutableNode::LoadCodeSection(dmlc::Stream* strm) {
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
  } else {
    return "%" + std::to_string(reg);
  }
}

std::string InstrArgToStr(InstrArg arg) {
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

std::string InstrArgToPyStr(InstrArg arg) {
  switch (arg.kind()) {
    case Instruction::kRegister:
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

String ExecutableNode::AsText() const {
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
          os << std::setw(6) << std::left << "call"
             << std::setw(16) << std::left << this->func_names[instr.func_idx]
             << " in: " << std::setw(12) << std::left
             << StrJoin<InstrArg>(instr.args, 0, instr.num_args, ", ", InstrArgToStr)
             << " dst: " << RegNameToStr(instr.dst) << "\n";
          break;
        }
        case Opcode::Ret: {
          os << std::setw(6) << std::left << "ret"
             << "ret " << RegNameToStr(instr.result) << "\n";
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

String ExecutableNode::AsPython() const {
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
            << StrJoin<InstrArg>(instr.args, 0, instr.num_args, ", ", InstrArgToPyStr) << "]";
          if (instr.dst != Instruction::kVoidArg)
            os << ", ret=ib.r(" << instr.dst << ")";
          os << ")\n";
          break;
        }
        case Opcode::Ret: {
          os << "    ib.emit_ret(ib.r(" << instr.result << "))\n";
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

TVM_REGISTER_GLOBAL("relax.Executable")
.set_body_typed([]() {
  return Executable();
});

TVM_REGISTER_GLOBAL("relax.ExecutableStats")
.set_body_typed([](Executable exec) {
  return exec->Stats();
});

TVM_REGISTER_GLOBAL("relax.ExecutableAsText")
.set_body_typed([](Executable exec) {
  return exec->AsText();
});

TVM_REGISTER_GLOBAL("relax.ExecutableAsPython").set_body_typed([](Executable exec) {
  return exec->AsPython();
});

TVM_REGISTER_GLOBAL("relax.ExecutableSaveToFile")
.set_body_typed([](Executable exec, std::string file_name) {
	return exec->SaveToFile(file_name);
});

TVM_REGISTER_GLOBAL("relax.ExecutableLoadFromFile")
.set_body_typed([](std::string file_name) {
	return ExecutableNode::LoadFromFile(file_name);
});

}  // namespace vm
}  // namespace relax
}  // namespace tvm
