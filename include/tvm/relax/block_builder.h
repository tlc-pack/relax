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
 * \file tvm/relax/block_builder.h
 * \brief The utility for constructing Relax binding blocks.
 */
#ifndef TVM_RELAX_BLOCK_BUILDER_H_
#define TVM_RELAX_BLOCK_BUILDER_H_

#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/utils.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>

#include <memory>
#include <stack>
#include <string>
#include <unordered_map>

namespace tvm {
namespace relax {

class BlockBuilder;

/*!
 * \brief A builder that provides APIs to build Relax binding blocks.
 */
class BlockBuilderNode : public Object {
 public:
  BlockBuilderNode();

  ~BlockBuilderNode();

  /*! \brief Begin to build a DataflowBlock. */
  void BeginDataflowBlock();

  /*! \brief Begin to build a BindingBlock. */
  void BeginBindingBlock();

  /*!
   * \brief End building a BindingBlock.
   * \return The BindingBlock being built.
   */
  BindingBlock EndBlock();

  /*!
   * \brief Check if the block being built is DataflowBlock or not.
   * \return A boolean that indicates if the block being built is DataflowBlock or not.
   */
  inline bool CurrentBlockIsDataFlow() { return CurrentFrame()->is_dataflow; }

  /*!
   * \brief Emits an Expr, and returns the variable it is bound to.
   * \param expr The Expr to be emitted.
   * \param name_hint Name hint for the bound variable.
   * \return The new variable that \p expr is bound to.
   */
  virtual Var Emit(const Expr& expr, std::string name_hint = "");

  /*!
   * \brief Emits a variable binding, and returns the bound Var.
   * \param binding The variable binding.
   * \return The bound variable.
   */
  virtual Var Emit(const VarBinding& binding);

  /*!
   * \brief Emit a MatchShape.
   * \param value The value of the MatchShape to be emitted.
   * \param pattern The pattern of the MatchShape to be emitted.
   * \param name_hint Name hint for the bound variable.
   * \return The variable bound to the MatchShape.
   */
  Var EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern, std::string name_hint = "");

  /*!
   * \brief Emit a MatchShape binding.
   * \param binding The MatchShape binding to be emitted.
   * \return The variable bound to the MatchShape.
   */
  Var EmitMatchShape(const MatchShape& binding);

  /*!
   * \brief Generate an output for the current dataflow block.
   * \param output The output variable of the block.
   * \param name_hint Name hint for the bound variable.
   * \return The variable bound to \p output.
   */
  Var EmitOutput(const Expr& output, std::string name_hint = "");

  /*!
   * \brief Generate an output for the current dataflow block.
   * \param binding The output binding to output.
   * \return The variable bound to \p output.
   */
  Var EmitOutput(const VarBinding& binding);

  /*!
   * \brief Lookup a var in the binding table \p binding_table_.
   * \param var The input var.
   * \return The Expr bound to the input \p var.
   * \note For function parameters, this function returns NullOpt.
   */
  Optional<Expr> LookupBinding(const Var& var);

  /*!
   * \brief Check if two shape expressions can be proven equal at compile time.
   * \param lhs The input lhs shape.
   * \param rhs The input rhs shape.
   * \return Whether we can prove lhs shape is the same as the rhs shape.
   */
  bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs);

  /*!
   * \brief Convert an expression to A-normal form, and try to eagerly infer types and shapes.
   * \param expr The input expression.
   * \return The normalized expression.
   */
  Expr Normalize(const Expr& expr);

  /*!
   * \brief Get the name table for generating unique names.
   *
   * \return The name table.
   */
  NameTable* name_table();

  /*!
   * \brief Add a Relax function or a TIR PrimFunc to \p context_mod_.
   * \param func The function to be added.
   * \param func_name_hint The name hint of the function to be added.
   * \note If the function to be added already exists in \p context_mod_, return its
   * GlobalVar directly.
   * \return The global var bound to the added function.
   */
  GlobalVar AddFuncToContext(const BaseFunc& func, const String& func_name_hint);

  /*!
   * \brief Get the context IRModule being built.
   * \return The IRModule being built by BlockBuilder.
   */
  IRModule GetContextIRModule() const;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.BlockBuilder";
  TVM_DECLARE_BASE_OBJECT_INFO(BlockBuilderNode, Object);

 private:
  Var Emit(const Expr& expr, bool is_dataflow, std::string name_hint);

  /*! \brief The IRModule being built by the BlockBuilder. */
  IRModule context_mod_;

  /*!
   * \brief A hashmap to store the mapping of Relax functions and TIR PrimFuncs
   * in \p _context_mod to their GlobalVar to avoid generating duplicated functions.
   */
  std::unordered_map<BaseFunc, GlobalVar, StructuralHash, StructuralEqual> func_map_;

 protected:
  /*!
   * \brief A representation of a block frame.
   *
   * A block frame is a record containing the bindings needed
   * to build a binding block, and a boolean to indicate if the
   * block being built is a DataflowBlock or not.
   */
  struct BlockFrame {
    Array<Binding> bindings;
    bool is_dataflow;
  };

  /*!
   * \brief Utility class for performing IR normalization (conversion to ANF, eager forward shape
   * and type inference).
   */
  class ExprNormalizer;

  friend class BlockBuilder;

  /*!
   * \brief Get the current block frame.
   * \return The current block frame.
   */
  BlockFrame* CurrentFrame();

  /*! \brief A stack to store block frames. */
  std::stack<BlockFrame> block_stack_;

  /*! \brief A diagnostic context for reporting errors. */
  DiagnosticContext diag_ctx_ = DiagnosticContext::Default(IRModule({}, {}));

  /*! \brief A binding table that maps var to value. */
  std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual> binding_table_;

  /*! \brief A name table to get unique names for IR construction. */
  std::unique_ptr<NameTable> name_table_;

  /*! \brief The internal normalizer used for ANF conversion. */
  std::unique_ptr<ExprNormalizer> normalizer_;
};

class BlockBuilder : public ObjectRef {
 public:
  /*!
   * \brief Create a BlockBuilder.
   * \return The created BlockBuilder.
   */
  TVM_DLL static BlockBuilder Create();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BlockBuilder, ObjectRef, BlockBuilderNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BLOCK_BUILDER_H_
