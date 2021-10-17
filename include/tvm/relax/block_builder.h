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
#include <tvm/relay/expr.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>

#include <memory>

namespace tvm {
namespace relax {

class BlockBuilder;

/*!
 * \brief Utility data structure for generating unique names for IR construction.
 */
class NameTable {
 public:
  /*!
   * \brief Generate a unique name with a specified prefix.
   * \param prefix The name prefix.
   * \return The generated name.
   */
  inline std::string GetUniqueName(std::string prefix) {
    std::replace(prefix.begin(), prefix.end(), '.', '_');
    std::string unique_prefix = prefix;
    auto it = alloc_map_.find(prefix);
    if (it != alloc_map_.end()) {
      while (alloc_map_.count(unique_prefix = prefix + std::to_string(++it->second)) > 0) {
      }
    }
    alloc_map_[unique_prefix] = 0;
    return unique_prefix;
  }

 private:
  std::unordered_map<std::string, uint32_t> alloc_map_;
};

/*!
 * \brief A builder that provides APIs to build Relax binding blocks.
 */
class BlockBuilderNode : public Object {
 public:
  BlockBuilderNode(std::shared_ptr<NameTable> name_table) : name_table_(name_table) {}

  ~BlockBuilderNode();

  BlockBuilderNode() { name_table_ = std::make_shared<NameTable>(); }

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
   * \brief Lookup a var in the binding table \p var_map_.
   * \param var The input var.
   * \return The Expr bound to the input \p var.
   */
  Expr LookupVar(const Var& var);
  /*!
   * \brief Check if two shape expressions can be proven equal at compile time.
   * \param lhs The input lhs shape.
   * \param rhs The input rhs shape.
   * \return Whether we can prove lhs shape is the same as the rhs shape.
   */
  bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs);
  /*!
   * \brief Normalize an Expr to complete its shape and type.
   * \param expr The input expr.
   * \return The expr with normalized shape and type.
   */
  Expr Normalize(const Expr& expr);
  /*!
   * \brief Create a BlockBuilder.
   * \return The created BlockBuilder.
   */
  TVM_DLL static BlockBuilder Create();

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.BlockBuilder";
  TVM_DECLARE_BASE_OBJECT_INFO(BlockBuilderNode, Object);

 private:
  Var Emit(const Expr& expr, bool is_dataflow, std::string name_hint);

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
  // TODO(@yuchen, @altanh): make var_map_ scoped, and decide if it should be in the builder
  std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual> var_map_;
  /*! \brief A name table to get unique names for IR construction. */
  std::shared_ptr<NameTable> name_table_;
};

class BlockBuilder : public ObjectRef {
 public:
  TVM_DLL explicit BlockBuilder(std::shared_ptr<NameTable> name_table);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BlockBuilder, ObjectRef, BlockBuilderNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BLOCK_BUILDER_H_
