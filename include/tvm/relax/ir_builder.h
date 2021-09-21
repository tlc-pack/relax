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
class LazyIRBuilder;

/*!
 * \brief The state of Relax function node being built.
 */
struct RelaxFunction {
  /*! \brief The function name. */
  Optional<GlobalVar> func_name = NullOpt;
  /*! \brief The function parameters. */
  Array<Var> params;
  /*! \brief The bindings in the function. */
  std::vector<Binding> bindings;
  /*! \brief The binding blocks in the function. */
  std::vector<BindingBlock> binding_blocks;
  /*! \brief The return of the function. */
  Expr ret = Tuple();
  /*! \brief The FunctionNode being built. */
  Function func;
};

/*!
 * \brief A builder that provides APIs to build Relax AST.
 */
class IRBuilderNode : public Object {
 public:
  /*!
   * \brief Fill the function name and parameters.
   */
  void FillFuncNameParam(const Array<Var>& params, const std::string& func_name);
  /*!
   * \brief Build a function node.
   */
  void BuildFunction();
  /*!
   * \brief Build a binding block.
   */
  virtual void BuildBlock();
  /*!
   * \brief Emit a call node.
   * \param call The CallNode to be emitted.
   * \return The variable being created and binded to \p call.
   */
  virtual Var Emit(const Call& call);
  /*!
   * \brief Emit a var binding.
   * \param binding The VarBinding to be emitted.
   * \return The VarNode of the VarBinding \p binding.
   */
  virtual Var Emit(const VarBinding& binding);
  /*!
   * \brief Emit a call node, and bind it to a Var.
   * \param var The VarNode to be binded with. \p var is reused implicitly.
   * \param call The CallNode to be emitted.
   * \return The VarNode to be binded with \p var.
   */
  virtual Var Emit(const Var& var, const Call& call);
  /*!
   * \brief Generate an output for the current dataflow block or function.
   * \param output The output variable of the block/function.
   * \return The variable being binded to \p output.
   */
  Var EmitOutput(const Expr& output);
  /*!
   * \brief Lookup a var in the binding table \p var_map_.
   */
  Expr LookupVar(const Var& var);
  /*!
   * \brief Get the function being built.
   */
  Function Get();
  /*!
   * \brief Get binding blocks being built.
   */
  std::vector<BindingBlock> GetBlocks();
  /*!
   * \brief Create a IRBuilder.
   * \return The created IRBuilder.
   */
  TVM_DLL static IRBuilder Create();

  /*! \brief A flag tracking if currently inside a dataflow block or not. */
  bool is_dataflow_ = false;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.IRBuilder";
  TVM_DECLARE_BASE_OBJECT_INFO(IRBuilderNode, Object);

 protected:
  /*! \brief The state of the function currently being built. */
  RelaxFunction func_;
  /*! \brief A global variable counter for naming global variables. */
  int global_var_counter_ = 0;
  /*! \brief A dataflow variable counter for naming dataflow variables. */
  int dataflow_var_counter_ = 0;
  /*! \brief A diagnostic context for reporting errors. */
  DiagnosticContext diag_ctx_ = DiagnosticContext::Default(IRModule({}, {}));
  /*! \brief A binding table that maps var to value. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> var_map_;
};

class IRBuilder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IRBuilder, ObjectRef, IRBuilderNode);
};

/*! \brief Auxiliary scope for building Relax function node,
 * similar to python's with syntax.
 *
 * \code
 * {
 *   With<FunctionScope> scope(ir_builder);
 *   // build function node.
 * }
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

/*! \brief Auxiliary scope for building Relax dataflow block,
 * similar to python's with syntax.
 *
 * \code
 * {
 *   With<DataflowScope> scope(ir_builder);
 *   // build dataflow block.
 * }
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

/*!
 * \brief A lazy builder to construct dataflow block in a copy-on-write fashion.
 */
class LazyIRBuilderNode : public IRBuilderNode {
 public:
  /*!
   * \brief Emit a call node in a copy-on-write way.
   * If no bindings in a dataflow block need to be rewritten, reuse the original variable instead of
   * emiting one. If any binding in the block needs to be rewritten, reconstruct the whole block
   * from scratch by emiting all previous bindings. 
   * \param call The CallNode to be emitted. 
   * \return The variable being created and binded to \p call.
   */
  virtual Var Emit(const Call& call);
  /*!
   * \brief Emit a var binding in a copy-on-write way.
   * \param binding The VarBinding to be emitted.
   * \return The VarNode of the VarBinding \p binding.
   */
  virtual Var Emit(const VarBinding& binding);
  /*!
   * \brief Emit a call node, and bind it to a Var in a copy-on-write way.
   * \param var The VarNode to be binded with.
   * \param call The CallNode to be emitted.
   * \return The VarNode to be binded with \p var.
   */
  virtual Var Emit(const Var& var, const Call& call);
  /*!
   * \brief Build a binding block.
   */
  virtual void BuildBlock();
  /*!
   * \brief Create a LazyIRBuilder.
   * \return The created LazyIRBuilder.
   */
  TVM_DLL static LazyIRBuilder Create(const DataflowBlock& block);

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.LazyIRBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(LazyIRBuilderNode, IRBuilderNode);

 private:
  /*! \brief Original DataflowBlock before rewriting. */
  DataflowBlock df_block_;
  /*! \brief index in the \p bindings. */
  int64_t index_ = 0;
  /*! \brief A flag tracking if current dataflow block needs to be rewritten or not. */
  bool is_rewrite_ = false;
};

class LazyIRBuilder : public IRBuilder {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LazyIRBuilder, IRBuilder, LazyIRBuilderNode);
};


}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_BUILDER_H_
