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

namespace tvm {
namespace relax {

using relay::Call;

class BlockBuilder;

/*!
 * \brief A builder that provides APIs to build Relax binding blocks.
 */
class BlockBuilderNode : public Object {
 public:
  void BeginBlock(bool is_dataflow);

  BindingBlock EndBlock();
  /*!
   * \brief Emit a Call, and return a newly created Var binded to the Call.
   * \param call The Call to be emitted.
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
   * \brief Emit a Call, and bind it to a Var.
   * \param var The Var to be binded with. \p var is reused implicitly if the shape 
   * and type of \p call matches \p var. Otherwise a new Var is created.
   * \param call The Call to be emitted.
   * \return The Var to be binded with \p var.
   */
  virtual Var Emit(const Var& var, const Call& call);
  /*!
   * \brief Emit a MatchShape.
   * \param value The value of the MatchShape to be emitted.
   * \param pattern The pattern of the MatchShape to be emitted.
   * \return The variable being binded to the MatchShape.
   */
  Var EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern);
  /*!
   * \brief Generate an output for the current dataflow block.
   * \param output The output variable of the block.
   * \return The variable being binded to \p output.
   */
  Var EmitOutput(const Var& var, const Expr& output);
  /*!
   * \brief Lookup a var in the binding table \p var_map_.
   */
  Expr LookupVar(const Var& var);

  /*!
   * \brief Check if two shape expressions can be proven equal at compile time.
   * \param lhs The input lhs shape.
   * \param rhs The input rhs shape.
   * \return Whether we can prove lhs shape == rhs shape.
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

 protected:
  struct BlockState {
    Array<Binding> bindings;
    bool is_dataflow;
  };

  /*! \brief  */
  std::stack<BlockState> block_stack_;
  /*! \brief A global variable counter for naming global variables. */
  int global_var_counter_ = 0;
  /*! \brief A dataflow variable counter for naming dataflow variables. */
  int dataflow_var_counter_ = 0;
  /*! \brief A diagnostic context for reporting errors. */
  DiagnosticContext diag_ctx_ = DiagnosticContext::Default(IRModule({}, {}));
  /*! \brief A binding table that maps var to value. */
  // TODO(@yuchen, @altanh): make var_map_ scoped, and decide if it should be in the builder 
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> var_map_;
};

class BlockBuilder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BlockBuilder, ObjectRef, BlockBuilderNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BLOCK_BUILDER_H_
