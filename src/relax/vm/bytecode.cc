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
 * \file src/relax/vm/bytecode.cc
 * \brief The bytecode for Relax virtual machine.
 */

#include <tvm/runtime/logging.h>
#include <tvm/relax/vm/bytecode.h>
#include <functional>

#include <sstream>

namespace tvm {
namespace relax {
namespace vm {

Instruction::Instruction() {}

Instruction::Instruction(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Call:
      this->func_idx = instr.func_idx;
      this->num_args = instr.num_args;
      this->args = instr.args;
      return;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

Instruction& Instruction::operator=(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Call:
      this->func_idx = instr.func_idx;
      this->num_args = instr.num_args;
      this->args = instr.args;
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
      return;
    default:
      std::ostringstream out;
      LOG(FATAL) << "Invalid instruction " << static_cast<int>(this->op);
  }
}

Instruction Instruction::Call(Index func_idx, Index num_args, 
                              ExecWord* args, RegName dst) {
  Instruction instr;
  instr.op = Opcode::Call;
  instr.dst = dst;
  instr.func_idx = func_idx;
  instr.num_args = num_args;
  instr.args = args;
  return instr;
}

}  // namespace vm
}  // namespace relax
}  // namespace tvm
