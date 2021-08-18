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
 * \file tvm/relax/ir_builder.h
 * \brief
 */
#ifndef TVM_RELAX_IR_BUILDER_H_
#define TVM_RELAX_IR_BUILDER_H_

#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

class IRBuilder;

/*!
 * \brief A representation of a Relax function.
 */
struct RelaxFunction {
  /*! \brief The bindings in the function. */
  std::vector<Binding> bindings;
  /*! \brief The return of the function. */
  Expr ret;
  /*! \brief The function body. */
  std::vector<BindingBlock> binding_blocks;
  /*! \brief The FunctionNode built in AST. */
  relax::Function func;
};

/*!
 * \brief A builder provides api to build Relax IR AST.
 */
class IRBuilderNode : public Object {
 public:
  /*!
   * \brief To annotate the start of a Relax function.
   * \param name The function name.
   * \param params The function parameters.
   */
  void BuildFunction(std::string name, Array<Var> params);

  void BuildBlock();

  Var Emit(relay::Call call);

  Var EmitDataflowOutput(Var var);
  /*!
   * \brief Emit outputs of a function.
   */
  void EmitOutput(Expr output);

  Function Get();

  inline void FlipState();

  TVM_DLL static IRBuilder Create();

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.IRBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRBuilderNode, Object);

 private:
  RelaxFunction func;
  bool is_dataflow = false;
  int global_var_counter = 0;
  int dataflow_var_counter = 0;
};

class IRBuilder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IRBuilder, ObjectRef, IRBuilderNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_BUILDER_H_
