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
 * \brief The utility for constructing Relax AST.
 */
#ifndef TVM_RELAX_IR_BUILDER_H_
#define TVM_RELAX_IR_BUILDER_H_

#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>

namespace tvm {
namespace relax {

using relay::Call;

class IRBuilder;

/*!
 * \brief A representation of a Relax function.
 */
struct RelaxFunction {
  /*! \brief The function name. */
  std::string func_name;
  /*! \brief The function parameters. */
  Array<Var> params;
  /*! \brief The bindings in the function. */
  std::vector<Binding> bindings;
  /*! \brief The binding blocks in the function. */
  std::vector<BindingBlock> binding_blocks;
  /*! \brief The return of the function. */
  Expr ret;
  /*! \brief The FunctionNode being built. */
  Function func;
};

/*!
 * \brief A builder that provides API to build Relax AST.
 */
class IRBuilderNode : public Object {
 public:
  /*!
   * \brief Fill the function name and parameters.
   */
  void FillFuncNameParam(const std::string& func_name, const Array<Var>& params);
  /*!
   * \brief Build a function node.
   */
  void BuildFunction();
  /*!
   * \brief Build a binding block.
   */
  void BuildBlock();
  /*!
   * \brief Emit a call node.
   * \param call The CallNode to be emitted.
   * \return The variable being binded to \p call.
   */
  Var Emit(const Call& call);
  /*!
   * \brief Emit a dataflow block's output variable.
   * \param var The output variable inside the dataflow block.
   * \return The variable being binded to the ouput \p var.
   */
  Var EmitDataflowOutput(const Var& var);
  /*!
   * \brief Emit a function's output variable.
   * \param output The output variable(s) of the function.
   */
  void EmitOutput(const Expr& output);
  /*!
   * \brief Get the function being built.
   */
  Function Get();
  /*!
   * \brief Get binding blocks being built.
   */
  std::vector<BindingBlock> GetBlocks();
  /*!
   * \brief Flip \p is_dataflow to indicate switching from DataflowBlock to BindingBlock or the
   * other way around.
   */
  inline void SwitchBlock();
  /*!
   * \brief Create a IRBuilder.
   * \return The IRBuilder.
   */
  TVM_DLL static IRBuilder Create();

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.IRBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRBuilderNode, Object);

 private:
  /*! \brief A representation of a function that stores states. */
  RelaxFunction func;
  /*! \brief A flag denoting inside a dataflow block or not. */
  bool is_dataflow = false;
  /*! \brief The global variable counter. */
  int global_var_counter = 0;
  /*! \brief The dataflow variable counter. */
  int dataflow_var_counter = 0;
};

class IRBuilder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IRBuilder, ObjectRef, IRBuilderNode);
};

/*! \brief Auxiliary scope for building function,
 * similar to python's with syntax.
 */
class FunctionScopeNode : public Object {
 public:
  IRBuilder ir_builder;
  void VisitAttrs(AttrVisitor* v) { v->Visit("ir_builder", &ir_builder); }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.FunctionScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionScopeNode, Object);
};

class FunctionScope : public ObjectRef {
 public:
  /*!
   * \brief construct from IRBuilder
   * \param ib The clauses in the specialized condition.
   */
  TVM_DLL FunctionScope(IRBuilder ib);
  TVM_DEFINE_OBJECT_REF_METHODS(FunctionScope, ObjectRef, FunctionScopeNode);
  class Internal;

 private:
  // Classes to get the Python `with` like syntax.
  friend class Internal;
  friend class With<FunctionScope>;
  // The entry of a function scope.
  TVM_DLL void EnterWithScope();
  // The exit of a function scope.
  TVM_DLL void ExitWithScope();
};

/*! \brief Auxiliary scope for building dataflow block,
 * similar to python's with syntax.
 */
class DataflowScopeNode : public Object {
 public:
  IRBuilder ir_builder;
  void VisitAttrs(AttrVisitor* v) { v->Visit("ir_builder", &ir_builder); }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.DataflowScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowScopeNode, Object);
};

class DataflowScope : public ObjectRef {
 public:
  TVM_DLL DataflowScope(IRBuilder ib);
  TVM_DEFINE_OBJECT_REF_METHODS(DataflowScope, ObjectRef, DataflowScopeNode);
  class Internal;

 private:
  // Classes to get the Python `with` like syntax.
  friend class Internal;
  friend class With<DataflowScope>;
  // The entry of a dataflow scope.
  TVM_DLL void EnterWithScope();
  // The exit of a dataflow scope.
  TVM_DLL void ExitWithScope();
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_BUILDER_H_
