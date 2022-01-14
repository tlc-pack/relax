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
 * \file tvm/relax/vm/exec_builder.h
 * \brief
 */
#ifndef TVM_RELAX_VM_EXEC_BUILDER_H_
#define TVM_RELAX_VM_EXEC_BUILDER_H_

#include <tvm/ir/expr.h>
#include <tvm/node/reflection.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include "./bytecode.h"
#include "./executable.h"

namespace tvm {
namespace relax {

namespace vm = tvm::runtime::relax_vm;

class ExecBuilder;

/*!
 * \brief A builder provides api to build VM executable with instructions.
 */
class ExecBuilderNode : public Object {
 public:
  /*! \brief The mutable internal executable node. */
  ObjectPtr<vm::ExecutableNode> exec;  // mutable
  /*!
   * \brief To annotate the start of a vm function.
   * \param func The function name.
   * \param num_inputs The number of inputs.
   */
  void EmitFunction(std::string func, int64_t num_inputs);
  /*!
   * \brief Emit a call instruction for a packed function.
   * \param func The packed function name.
   * \param args The arguments of the function.
   * \param ret The return register.
   */
  void EmitCall(std::string func, std::vector<vm::Instruction::Arg> args, vm::RegName ret);
  /*!
   * \brief Emit a ret instruction.
   * \param result The return result.
   */
  void EmitRet(vm::RegName result);
  /*!
   * \brief Emit a goto instruction.
   * \param pc_offset The program counter offset as the jump offset.
   */
  void EmitGoto(vm::Index pc_offset);
  /*!
   * \brief Emit an If instruction.
   * \param test The register containing the test value.
   * \param target The register containing the target value.
   * \param true_offset The program counter offset for the true branch.
   * \param false_offset The program counter offset for the false branch.
   */
  void EmitIf(vm::RegName test, vm::RegName target, vm::Index true_offset, vm::Index false_offset);
  /*!
   * \brief Emit a constant value to the constant pool.
   * \return The index that represents the constant.
   */
  vm::Index EmitConstant(TVMRetValue obj);
  /*!
   * \brief Get the built executable.
   * \return The built executable.
   */
  vm::Executable Get();
  /*!
   * \brief Create a ExecBuilder.
   * \return The ExecBuilder.
   */
  TVM_DLL static ExecBuilder Create();

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.ExecBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecBuilderNode, Object);

 private:
  /*!
   * \brief Formalize the executable.
   */
  void Formalize();
};

class ExecBuilder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ExecBuilder, ObjectRef, ExecBuilderNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_EXEC_BUILDER_H_
