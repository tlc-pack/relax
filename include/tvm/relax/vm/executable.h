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
 * \file tvm/relax/vm/executable.h
 * \brief
 */
#ifndef TVM_RELAX_VM_EXECUTABLE_H_
#define TVM_RELAX_VM_EXECUTABLE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "./bytecode.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

class Executable;

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
};

/*!
 * \brief The executable emitted by the VM compiler.
 *
 * The executable contains information (e.g. data in different memory regions)
 * to run in a virtual machine.
 */
class ExecutableNode : public Object {
 public:
  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globls and constants, etc.
   */
  std::string Stats() const;
  /*!
   * \brief Get the i-th instruction from the executable.
   * \return The instruction.
   */
  Instruction GetInstruction(Index i) const;
  /*!
   * \brief Print the instructions as text format.
   */
  String AsText() const;
  /*!
   * \brief Print the instructions as python program.
   */
  String AsPython() const;
  /*!
   * \brief Write the Executable to the binary stream in serialized form.
   * \param stream The binary stream to save the executable to.
   */
  void SaveToBinary(dmlc::Stream* stream);
  /*!
   * \brief Load Executable from the binary stream in serialized form.
   * \param stream The binary stream that load the executable from.
   */
  static Executable LoadFromBinary(void* stream);
  /*!
   * \brief Write the Executable to the provided path as a file contianing its serialized content.
   * \param path The path to write the serialized data to.
   */
  void SaveToFile(const std::string& path);
  /*!
   * \brief Load Executable from the file.
   * \param file_name The file that load the executable from.
   */
  static Executable LoadFromFile(const std::string& file_name);
  /*! \brief The virtual machine's function table. */
  std::vector<VMFunction> global_funcs;
  /*! \brief A map from globals (as strings) to their index in the function map. */
  std::unordered_map<std::string, Index> global_map;
  /*! \brief The global constant pool. */
  std::vector<TVMRetValue> constants;
  /*! \brief The name of packed functions. */
  std::vector<std::string> func_names;
  /*! \brief A mapping from the packed function (as string) to the index that
   * corresponds to the position of the `packed_funcs` list in a `VirtualMachine` object.
   */
  std::unordered_map<std::string, Index> func2idx;
  /*! \brief The offset of instruction. */
  std::vector<Index> instr_offset;
  /*! \brief The byte data of instruction. */
  std::vector<ExecWord> instr_data;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Executable";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutableNode, Object);

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

/*! \brief Reference to Executable. */
class Executable : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Executable, ObjectRef, ExecutableNode);
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RELAX_VM_EXECUTABLE_H_
