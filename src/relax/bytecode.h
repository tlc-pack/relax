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

/*! \brief An enumeration of Relay's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
 */
enum class Opcode {
  CallPacked = 1U,
  // Move = 0U,
  // Ret = 1U,
  // Invoke = 2U,
  // InvokeClosure = 3U,
  // InvokePacked = 4U,
  // AllocTensor = 5U,
  // AllocTensorReg = 6U,
  // AllocADT = 7U,
  // AllocClosure = 8U,
  // GetField = 9U,
  // If = 10U,
  // LoadConst = 11U,
  // Goto = 12U,
  // GetTag = 13U,
  // LoadConsti = 14U,
  // Fatal = 15U,
  // AllocStorage = 16U,
  // ShapeOf = 17U,
  // ReshapeTensor = 18U,
  // DeviceCopy = 19U,
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
  /*! \brief The instruction opcode. */
  Opcode op;

  /*! \brief The destination register. */
  RegName dst;

  union {
    struct /* CallPacked */ {
      /*! \brief The index into the packed function table. */
      Index packed_index;
      /*! \brief The number of arguments to the packed function. */
      Index num_args;
      /*! \brief The registers containing the arguments. */
      RegName* args;
    };

    // struct /* InvokeClosure Operands */ {
    //   /*! \brief The register containing the closure. */
    //   RegName closure;
    //   /*! \brief The number of arguments to the closure. */
    //   Index num_closure_args;
    //   /*! \brief The closure arguments as an array. */
    //   RegName* closure_args;
    // };
    // struct /* InvokePacked Operands */ {
    //   /*! \brief The index into the packed function table. */
    //   Index packed_index;
    //   /*! \brief The arity of the packed function. */
    //   Index arity;
    //   /*! \brief The number of outputs produced by the packed function. */
    //   Index output_size;
    //   /*! \brief The arguments to pass to the packed function. */
    //   RegName* packed_args;
    // };
    // struct /* Invoke Operands */ {
    //   /*! \brief The function to call. */
    //   Index func_index;
    //   /*! \brief The number of arguments to the function. */
    //   Index num_args;
    //   /*! \brief The registers containing the arguments. */
    //   RegName* invoke_args_registers;
    // };
  };

  static Instruction CallPacked(Index index, Index num_args,
                                const std::vector<RegName>& args);
  // /*!
  //  * \brief Construct a invoke packed instruction.
  //  * \param packed_index The index of the packed function.
  //  * \param arity The arity of the function.
  //  * \param output_size The number of outputs of the packed function.
  //  * \param args The argument registers.
  //  * \return The invoke packed instruction.
  //  */
  // static Instruction InvokePacked(Index packed_index, Index arity, Index output_size,
  //                                 const std::vector<RegName>& args);
  // /*!
  //  * \brief Construct an invoke instruction.
  //  * \param func_index The index of the function to invoke.
  //  * \param args The registers containing the arguments.
  //  * \param dst The destination register.
  //  * \return The invoke instruction.
  //  */
  // static Instruction Invoke(Index func_index, const std::vector<RegName>& args, RegName dst);
  // /*!
  //  * \brief Construct an invoke closure instruction.
  //  * \param closure The register of the closure to invoke.
  //  * \param args The registers containing the arguments.
  //  * \param dst The destination register.
  //  * \return The invoke closure instruction.
  //  */
  // static Instruction InvokeClosure(RegName closure, const std::vector<RegName>& args, RegName dst);

  Instruction();
  Instruction(const Instruction& instr);
  Instruction& operator=(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NEW_VM_BYTECODE_H_
