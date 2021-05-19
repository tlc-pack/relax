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
 * \file tvm/runtime/vm/bytecode.h
 * \brief The bytecode for the virtual machine.
 */
#ifndef TVM_RUNTIME_NEW_VM_BYTECODE_H_
#define TVM_RUNTIME_NEW_VM_BYTECODE_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace runtime {
namespace new_vm {


/*! \brief A register name. */
using RegName = int64_t;

/*! \brief An alias for the integer type used ubiquitously
 * in the VM.
 */
using Index = int64_t;

enum ArgKind {
  kRegister = 0,
  kImmediate = 1,
  kConstIdx = 2,
};

constexpr int64_t kVoidArg = 0xFE0201975A;

struct InstrArg {
  explicit InstrArg() : data(kVoidArg) {}
  explicit InstrArg(int64_t data) : data(data) {}
  InstrArg(ArgKind kind, Index value) {
    // TODO(ziheng): check value
    this->data = (uint64_t(kind) << 56) | (value & ((uint64_t(1) << 56)  - 1));
  }
  ArgKind kind() {
    uint8_t kind = (data >> 56) & 0xFF;
    return ArgKind(kind);
  }
  int64_t value() {
    return data & ((int64_t(1) << 56) - 1);
  }
  int64_t data;
};

/*! \brief An enumeration of Relay's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
 */
enum class Opcode {
  Call = 1U,
};

/*! \brief A single virtual machine instruction.
 *
 * The representation of the instruction is as
 * a tagged union.
 *
 * The first field represents which instruction,
 * and by extension which field of the union
 * is active.
 */

// // option1
// class ByteCode {
//   std::vector<int64_t> instr;
//   std::vector<int64_t> offset;
//   
//   Instruction GetInstr(index);
// }
// // option2
// std::vector<int64_t> instr;
// std::vector<int64_t> instr_args;

struct Instruction {
  /*! \brief The instruction opcode. */
  Opcode op;

  /*! \brief The destination register. */
  RegName dst; 

  union {
    struct /* CallPacked */ {
      /*! \brief The index into the packed function table. */
      Index func_index;
      /*! \brief The number of arguments to the packed function. */
      Index num_args;

      Index arg_index;
    };
  };

  static Instruction Call(Index func_index, Index num_args,
                          Index arg_index,
                          RegName dst);

  Instruction();
  Instruction(const Instruction& instr);
  Instruction& operator=(const Instruction& instr);
  ~Instruction();
};

}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NEW_VM_BYTECODE_H_
