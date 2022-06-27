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
 * \file tvm/runtime/relax_vm/executable.h
 */
#ifndef TVM_RUNTIME_RELAX_VM_EXECUTABLE_H_
#define TVM_RUNTIME_RELAX_VM_EXECUTABLE_H_

#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "./bytecode.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief An object representing a vm closure.
 */
class VMClosureObj : public ClosureObj {
 public:
  /*!
   * \brief The function name. The function could be any
   * function object that is compatible to the VM runtime.
   */
  String func_name;
  /*! \brief The free variables of the closure. */
  Array<ObjectRef> free_vars;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(VMClosureObj, ClosureObj);
};

/*! \brief reference to closure. */
class VMClosure : public Closure {
 public:
  VMClosure(String func_name, Array<ObjectRef> free_vars);
  TVM_DEFINE_OBJECT_REF_METHODS(VMClosure, Closure, VMClosureObj);
};

/*!
 * \brief A representation of a Relax function in the VM.
 *
 * Contains metadata about the compiled function, as
 * well as the compiled VM instructions.
 */
struct VMFunction {
  /*! \brief The function's name. */
  std::string name;
  /*! \brief The start instruction index of the function. */
  Index start_instr;
  /*! \brief The number of arguments of the function. */
  Index num_args;
  /*! \brief The register file size of the function. */
  Index register_file_size;
  /*! \brief The function parameter names.*/
  std::vector<std::string> param_names;
};

/*!
 * \brief The executable emitted by the VM compiler.
 *
 * The executable contains information (e.g. data in different memory regions)
 * to run in a virtual machine.
 */
class Executable : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get a PackedFunc from the executable module.
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   * \return PackedFunc or nullptr when it is not available.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;
  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globals and constants, etc.
   * \return The statistics represented by a string.
   */
  std::string Stats() const;
  /*!
   * \brief Get the i-th instruction from the executable.
   * \param i The index of the instruction to be fetched.
   * \return The instruction.
   */
  Instruction GetInstruction(Index i) const;
  /*!
   * \brief Set j-th byte data of i-th instruction to val.
   * \param i The index of the instruction to be updated.
   * \param j The index of the byte data of the instruction to be updated.
   * \param val The value to be set
   */
  void SetInstructionData(Index i, Index j, ExecWord val);
  /*!
   * \brief Print the instructions as text format.
   * \return The text format of the instructions.
   */
  String AsText() const;
  /*!
   * \brief Print the instructions as python program.
   * \return The python program of the instructions, represented by a string.
   */
  String AsPython() const;
  /*!
   * \brief Write the Executable to the binary stream in serialized form.
   * \param stream The binary stream to save the executable to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;
  /*!
   * \brief Load Executable from the binary stream in serialized form.
   * \param stream The binary stream that load the executable from.
   * \return The loaded executable, in the form of a `runtime::Module`.
   */
  static Module LoadFromBinary(void* stream);
  /*!
   * \brief Write the Executable to the provided path as a file containing its serialized content.
   * \param file_name The name of the file to write the serialized data to.
   * \param format The target format of the saved file.
   */
  void SaveToFile(const std::string& file_name, const std::string& format) final;
  /*!
   * \brief Load Executable from the file.
   * \param file_name The path of the file that load the executable from.
   * \return The loaded executable, in the form of a `runtime::Module`.
   */
  static Module LoadFromFile(const std::string& file_name);

  /*! \brief The virtual machine's function table. */
  std::vector<VMFunction> global_funcs;
  /*! \brief A map from globals (as strings) to their index in the function map. */
  std::unordered_map<std::string, Index> global_map;
  /*! \brief The global constant pool. */
  std::vector<TVMRetValue> constants;
  /*! \brief The name of packed functions. */
  std::vector<std::string> func_names;
  /*!
   * \brief A mapping from the packed function (as string) to the index that
   * corresponds to the position of the `packed_funcs` list in a `VirtualMachine` object.
   */
  std::unordered_map<std::string, Index> func2idx;
  /*! \brief The offset of instruction. */
  std::vector<Index> instr_offset;
  /*! \brief The byte data of instruction. */
  std::vector<ExecWord> instr_data;

  virtual ~Executable() {}

  const char* type_key() const final { return "relax.Executable"; }

 private:
  /*!
   * \brief Save the globals.
   * \param strm The input stream.
   */
  void SaveGlobalSection(dmlc::Stream* strm);
  /*!
   * \brief Save the constant pool.
   * \param strm The input stream.
   */
  void SaveConstantSection(dmlc::Stream* strm);
  /*!
   * \brief Save the instructions.
   * \param strm The input stream.
   */
  void SaveCodeSection(dmlc::Stream* strm);
  /*!
   * \brief Save the packed functions.
   * \param strm The input stream.
   */
  void SavePackedFuncNames(dmlc::Stream* strm);
  /*!
   * \brief Load the globals.
   * \param strm The input stream.
   */
  void LoadGlobalSection(dmlc::Stream* strm);
  /*!
   * \brief Load the constant pool.
   * \param strm The input stream.
   */
  void LoadConstantSection(dmlc::Stream* strm);
  /*!
   * \brief Load the instructions.
   * \param strm The input stream.
   */
  void LoadCodeSection(dmlc::Stream* strm);
  /*!
   * \brief Save the packed functions.
   * \param strm The input stream.
   */
  void LoadPackedFuncNames(dmlc::Stream* strm);
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_EXECUTABLE_H_
