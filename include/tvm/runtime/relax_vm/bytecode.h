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
 * \file tvm/runtime/relax_vm/bytecode.h
 * \brief The bytecode for the virtual machine.
 */
#ifndef TVM_RUNTIME_RELAX_VM_BYTECODE_H_
#define TVM_RUNTIME_RELAX_VM_BYTECODE_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief The storage type for the bytecode in the VM.
 */
using ExecWord = int64_t;

/*! \brief A register name. */
using RegName = ExecWord;

/*!
 * \brief An alias for the integer type used ubiquitously in the VM.
 */
using Index = ExecWord;

/*!
 * \brief An enumeration of Relax's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
 */
enum class Opcode {
  Call = 1U,
  Ret = 2U,
  Goto = 3U,
  If = 4U,
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
  /*! \brief Random magic number that represents void argument. */
  static constexpr RegName kVoidArg = 0x00EC66FE0321975A;
  /*! \brief Random magic number that represents the VM. */
  static constexpr RegName kVMRegister = 0x008D14FA4379015C;
  /*!
   * \brief The kind of instruction's argument.
   */
  enum ArgKind {
    kRegister = 0,
    kImmediate = 1,
    kConstIdx = 2,
  };
  /*!
   * \brief The auxiliary data structure for instruction argument.
   */
  struct Arg {
    /*! \brief The number of bit for storing value. */
    static constexpr ExecWord kValueBit = sizeof(ExecWord) * 8 - 8;
    /*! \brief The bit mask of the value part. */
    static constexpr ExecWord kValueMask = (static_cast<ExecWord>(1) << kValueBit) - 1;
    /*! \brief Construct a void argument. */
    Arg() : data(Instruction::kVoidArg) {}
    /*! \brief Construct from the data. */
    explicit Arg(ExecWord data) : data(data) {}
    /*! \brief Construct from the kind and value. */
    Arg(ArgKind kind, Index value) {
      // TODO(ziheng): check value?
      this->data = (static_cast<ExecWord>(kind) << kValueBit) | (value & kValueMask);
    }
    /*!
     * \brief Get the kind of argument..
     * \return The kind of argument.
     */
    ArgKind kind() const {
      uint8_t kind = (data >> kValueBit) & 0xFF;
      return Instruction::ArgKind(kind);
    }
    /*!
     * \brief Get the value of argument..
     * \return The value of argument.
     */
    ExecWord value() const { return data & ((static_cast<ExecWord>(1) << kValueBit) - 1); }
    /*! \brief The underlying stored data. */
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
      /*! \brief The arguments of the packed function. */
      Arg* args;
    };
    struct /* Ret */ {
      /*! \brief The return result. */
      RegName result;
    };
    struct /* Goto */ {
      /*! \brief The jump offset. */
      Index pc_offset;
    };
    struct /* If */ {
      /*! \brief The register containing the cond value. */
      RegName cond;
      /*! \brief The program counter offset for the false branch. */
      Index false_offset;
    };
  };
  /*!
   * \brief Construct a Call instruction.
   * \param func_idx The index of the function to call.
   * \param num_args The number of arguments.
   * \param args The input arguments.
   * \param dst The destination register.
   * \return The call instruction.
   */
  static Instruction Call(Index func_idx, Index num_args, Arg* args, RegName dst);
  /*!
   * \brief Construct a return instruction.
   * \param result The register containing the return value.
   * \return The return instruction.
   */
  static Instruction Ret(RegName result);
  /*!
   * \brief Construct a goto instruction.
   * \param pc_offset The register containing the jump offset.
   * \return The goto instruction.
   */
  static Instruction Goto(RegName pc_offset);
  /*!
   * \brief Construct an If instruction.
   * \param cond The register containing the cond value.
   * \param false_offset The program counter offset for the false branch.
   * \return The If instruction.
   */
  static Instruction If(RegName cond, Index false_offset);
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_BYTECODE_H_
