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
 * \file tvm/relax/vm/bytecode.h
 * \brief The bytecode for the virtual machine.
 */
#ifndef TVM_RELAX_VM_BYTECODE_H_
#define TVM_RELAX_VM_BYTECODE_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {
namespace vm {


/*! \brief A register name. */
using RegName = int64_t;

/*! \brief An alias for the integer type used ubiquitously
 * in the VM.
 */
using Index = int64_t;

using ExecWord = int64_t; 

/*! \brief An enumeration of Relay's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
 */
enum class Opcode {
  Call = 1U,
  Ret = 2U,
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

struct Instruction {
  static constexpr ExecWord kVoidArg = 0xFE0321975A;

  enum ArgKind {
    kRegister = 0,
    kImmediate = 1,
    kConstIdx = 2,
  };

  struct Arg {
    explicit Arg() : data(Instruction::kVoidArg) {}
    explicit Arg(ExecWord data) : data(data) {}
    Arg(ArgKind kind, Index value) {
      // TODO(ziheng): check value
      size_t value_bit = sizeof(ExecWord) * 8 - 8;
      this->data = (static_cast<ExecWord>(kind) << value_bit) |
                   (value & ((static_cast<ExecWord>(1) << value_bit)  - 1));
    }
    ArgKind kind() {
      size_t value_bit = sizeof(ExecWord) * 8 - 8;
      uint8_t kind = (data >> value_bit) & 0xFF;
      return Instruction::ArgKind(kind);
    }
    ExecWord value() {
      size_t value_bit = sizeof(ExecWord) * 8 - 8;
      return data & ((static_cast<ExecWord>(1) << value_bit) - 1);
    }
    ExecWord data;
  };


  /*! \brief The instruction opcode. */
  Opcode op;

  /*! \brief The destination register. */
  RegName dst; 

  union {
    struct /* Call */ {
      /*! \brief The index into the packed function table. */
      Index func_idx;
      /*! \brief The number of arguments to the packed function. */
      Index num_args;

      Arg* args;
    };

    struct /* Ret */ {
      RegName result;
    };
  };

  static Instruction Call(Index func_idx, Index num_args,
                          Arg* args,
                          RegName dst);

  static Instruction Ret(RegName result);
};

using InstrArg = Instruction::Arg;

}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_BYTECODE_H_
