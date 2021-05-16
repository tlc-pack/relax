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
 * \file src/runtime/vm/bytecode.cc
 * \brief The bytecode for Relay virtual machine.
 */

#include <tvm/runtime/logging.h>
#include <functional>
#include "./bytecode.h"

#include <sstream>

namespace tvm {
namespace runtime {
namespace new_vm {

Instruction::Instruction() {}

template <typename T>
static T* Duplicate(T* src, Index size) {
  auto dst = new T[size];
  std::copy(src, src + size, dst);
  return dst;
}

Instruction::Instruction(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Call:
      this->packed_index = instr.packed_index;
      this->num_args = instr.num_args;
      this->args = Duplicate<InstrArg>(instr.args, instr.num_args);
      return;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

template <typename T>
static inline void FreeIf(T* t) {
  if (t != nullptr) {
    delete t;
  }
}

Instruction& Instruction::operator=(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Call:
      this->packed_index = instr.packed_index;
      this->num_args = instr.num_args;
      FreeIf(this->args);
      this->args = Duplicate<InstrArg>(instr.args, instr.num_args);
      return *this;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

Instruction::~Instruction() {
  switch (this->op) {
    case Opcode::Call:
      delete[] this->args;
      return;
    default:
      std::ostringstream out;
      LOG(FATAL) << "Invalid instruction " << static_cast<int>(this->op);
  }
}

Instruction Instruction::Call(Index packed_index, Index num_args, 
                              const std::vector<InstrArg>& args,
                              RegName dst) {
  Instruction instr;
  instr.op = Opcode::Call;
  instr.dst = dst;
  instr.packed_index = packed_index;
  instr.num_args = num_args;
  instr.args = new InstrArg[num_args];
  for (Index i = 0; i < num_args; ++i) {
    instr.args[i] = args[i];
  }
  return instr;
}

void DLDatatypePrint(std::ostream& os, const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      os << "int";
      break;
    case kDLUInt:
      os << "uint";
      break;
    case kDLFloat:
      os << "float";
      break;
  }

  os << int(dtype.bits);
  if (dtype.lanes != 1) {
    os << "x" << dtype.lanes;
  }
}

template <typename T>
std::string StrJoin(T* items, int offset, int cnt,
                    std::string delim = ", ",
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
  if (reg == kVoidArg) {
    return "void";
  } else {
    return "%" + std::to_string(reg);
  }

}

std::string InstrArgToStr(InstrArg arg) {
  switch(arg.kind()) {
    case kRegister:
      return RegNameToStr(arg.value());
    case kImmediate:
      return "i" + std::to_string(arg.value());
    case kConstIdx:
      return "c[" + std::to_string(arg.value()) + "]";
    default:
      LOG(FATAL) << "Wrong instruction kind: " << arg.kind();
      return "";
  }
}

void InstructionPrint(std::ostream& os, const Instruction& instr) {
  switch (instr.op) {
    case Opcode::Call: {
      os << "call PackedFunc[" << instr.packed_index << "] ["
         << StrJoin<InstrArg>(instr.args, 0, instr.num_args, ", ", InstrArgToStr)
         << "] ret " << RegNameToStr(instr.dst);
      break;
    }
    default:
      LOG(FATAL) << "should never hit this case" << static_cast<int>(instr.op);
      break;
  }
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  InstructionPrint(os, instr);
  return os;
}

}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
