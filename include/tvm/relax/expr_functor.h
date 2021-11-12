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
 * \file tvm/relax/expr_functor.h
 * \brief A more powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#ifndef TVM_RELAX_EXPR_FUNCTOR_H_
#define TVM_RELAX_EXPR_FUNCTOR_H_

#include <tvm/ir/error.h>
#include <tvm/node/functor.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>

#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tvm {
namespace relax {

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 * \sa tvm/ir_functor.h
 *
 * \tparam FType function signiture
 *  This type is only defined for FType with function signature R(const Expr&,
 * Args...)
 */
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT \
  { return VisitExprDefault_(op, std::forward<Args>(args)...); }

#define RELAX_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~ExprFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Expr& n, Args... args) { return VisitExpr(n, std::forward<Args>(args)...); }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const Expr& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitExpr_(const ConstantNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const VarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DataflowVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ShapeExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ExternFuncNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GlobalVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SeqExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IfNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleGetItemNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAX_EXPR_FUNCTOR_DISPATCH(ConstantNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(TupleNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(VarNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(DataflowVarNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(ShapeExprNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(ExternFuncNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(GlobalVarNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(FunctionNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(CallNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(SeqExprNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(IfNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(OpNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(TupleGetItemNode);
    return vtable;
  }
};


template <typename FType>
class BindingFunctor;

// functions to be overriden.
#define BINDING_FUNCTOR_DEFAULT \
  { return VisitBindingDefault_(op, std::forward<Args>(args)...); }

#define RELAX_BINDING_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitBinding_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class BindingFunctor<R(const Binding& n, Args...)> {
 private:
  using TSelf = BindingFunctor<R(const Binding& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~BindingFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The binding node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Binding& n, Args... args) { return VisitBinding(n, std::forward<Args>(args)...); }
  /*!
   * \brief The functor call.
   * \param n The binding node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitBinding(const Binding& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitBinding_(const VarBindingNode* op, Args... args) BINDING_FUNCTOR_DEFAULT;
  virtual R VisitBinding_(const MatchShapeNode* op, Args... args) BINDING_FUNCTOR_DEFAULT;
  virtual R VisitBindingDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAX_BINDING_FUNCTOR_DISPATCH(VarBindingNode);
    RELAX_BINDING_FUNCTOR_DISPATCH(MatchShapeNode);
    return vtable;
  }
};


template <typename FType>
class BindingBlockFunctor;

// functions to be overriden.
#define BINDINGBLOCK_FUNCTOR_DEFAULT \
  { return VisitBindingBlockDefault_(op, std::forward<Args>(args)...); }

#define RELAX_BINDINGBLOCK_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitBindingBlock_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class BindingBlockFunctor<R(const BindingBlock& n, Args...)> {
 private:
  using TSelf = BindingBlockFunctor<R(const BindingBlock& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~BindingBlockFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The BindingBlock node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const BindingBlock& n, Args... args) { return VisitBindingBlock(n, std::forward<Args>(args)...); }
  /*!
   * \brief The functor call.
   * \param n The BindingBlock node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitBindingBlock(const BindingBlock& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitBindingBlock_(const BindingBlockNode* op, Args... args) BINDINGBLOCK_FUNCTOR_DEFAULT;
  virtual R VisitBindingBlock_(const DataflowBlockNode* op, Args... args) BINDINGBLOCK_FUNCTOR_DEFAULT;
  virtual R VisitBindingBlockDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAX_BINDINGBLOCK_FUNCTOR_DISPATCH(BindingBlockNode);
    RELAX_BINDINGBLOCK_FUNCTOR_DISPATCH(DataflowBlockNode);
    return vtable;
  }
};

/*!
 * \brief A simple visitor wrapper around ExprFunctor.
 *  Recursively visit the content.
 */
class ExprVisitor : public ExprFunctor<void(const Expr&)>,
                    public BindingFunctor<void(const Binding&)>,
                    public BindingBlockFunctor<void(const BindingBlock&)> {
 public:
  /*!
   * \brief Generic dispatcher for Expr.
   * \param expr The expr to be visited.
   */
  void VisitExpr(const Expr& expr) override;
  // specific leaf level visitor functions
  void VisitExpr_(const ConstantNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const DataflowVarNode* op) override;
  void VisitExpr_(const ShapeExprNode* op) override;
  void VisitExpr_(const ExternFuncNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const SeqExprNode* op) override;
  void VisitExpr_(const IfNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  void VisitBinding(const Binding& binding) override;
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingNode* binding);
  virtual void VisitBinding_(const MatchShapeNode* binding);

  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   */
  void VisitBindingBlock(const BindingBlock& block) override;
  // specific leaf level visitor functions
  virtual void VisitBindingBlock_(const BindingBlockNode* block);
  virtual void VisitBindingBlock_(const DataflowBlockNode* block);

  virtual void VisitType(const Type& t);
  virtual void VisitSpan(const Span& span);  
};

void PostOrderVisit(const Expr& node, std::function<void(const Expr&)> fvisit);

/*!
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator treats Expr as dataflow graph, and only Mutate each Expr once.
 * The mutated results are memoized in a map and reused so that
 * local transformation on the dataflow preserves the graph structure.
 */
class ExprMutator : public ExprFunctor<Expr(const Expr&)>,
                    public BindingFunctor<void(const Binding&)>,
                    public BindingBlockFunctor<BindingBlock(const BindingBlock&)> {
 public:
  ExprMutator() {
    builder_ = BlockBuilder::Create();
  }

  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const DataflowVarNode* op) override;
  Expr VisitExpr_(const ShapeExprNode* op) override;
  Expr VisitExpr_(const ExternFuncNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* op) override;
  Expr VisitExpr_(const SeqExprNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;

  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  void VisitBinding(const Binding& binding) override;
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingNode* binding);
  virtual void VisitBinding_(const MatchShapeNode* binding);

  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  BindingBlock VisitBindingBlock(const BindingBlock& block) override;
  // specific leaf level visitor functions
  virtual BindingBlock VisitBindingBlock_(const BindingBlockNode* block);
  virtual BindingBlock VisitBindingBlock_(const DataflowBlockNode* block);

  /*!
   * \brief Rewrite the var definition site.
   * \param var The var to be visited.
   * \return The var after post-order rewritten.
   * \note VisitExpr_(const VarNode*) will only visit the usage site of an Var
   */
  virtual Var VisitVarDef(const Var& var);

 protected:
  class ExprNormalizer;

  /*!
   * \brief Rewrite the expr with a new scope, used in a Function's body and the branches of If.
   * \param expr The expr to be visited.
   * \return The expr after visiting.
   */
  Expr VisitWithNewScope(const Expr& expr);

  /*!
   * \brief Look up the value bound to a variable.
   * \param var The var to be looked up.
   * \return The value bound to the input \p var.
   */
  Expr LookupBinding(const Var& var);

  /*!
   * \brief Post-order rewrite a node and normalize.
   * \param T The node type to be rewritten.
   * \param op The node to be rewritten.
   * \return The node after post rewritten.
   */
  template <typename T>
  Expr VisitExprPostOrder_(const T* op) {
    return builder_->Normalize(ExprMutator::VisitExpr_(op));
  }

  /*!
   * \brief Create a new var with specified shape and type if it's original shape or type does not
   * match with the specified ones.
   * \param var The var to be updated.
   * \param shape The specified shape.
   * \param type The specified type.
   * \return The var filled with \p shape and \p type.
   */
  Var WithShapeAndType(Var var, Optional<ObjectRef> shape, Type type);

  /*! \brief Internal block builder to emit bindings during rewriting. */
  BlockBuilder builder_;

  /*! \brief Remap a var to a new var in use-site. */
  std::unordered_map<Id, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
};

// TODO(@yuchen, @altan): Refactor to enforce dataflow mutator only rewrite stuff in dataflow blocks
/*! \brief Dataflow Graph Rewriting for Custom Rewriting Passes
 */
// class DataflowMutator : public ExprMutator {
//  public:
//   void VisitBinding(const Binding& binding) final;

//   virtual void VisitDataflowVarBinding(const VarBinding& binding);
// };

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_EXPR_FUNCTOR_H_
