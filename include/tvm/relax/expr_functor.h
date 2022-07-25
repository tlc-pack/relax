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

#define PY_EXPR_VISITOR_DEFAULT(N, NAME, DEFAULT) \
  {                                               \
    auto it = operator->()->map_.find(NAME);      \
    if (it != operator->()->map_.end())           \
      it->second(N);                              \
    else                                          \
      DEFAULT;                                    \
  }

#define PY_EXPR_MUTATOR_DEFAULT(N, NAME, DEFAULT, RET_TYPE) \
  {                                                         \
    auto it = operator->()->map_.find(NAME);                \
    if (it != operator->()->map_.end()) {                   \
      RET_TYPE ret = it->second(N);                         \
      return ret;                                           \
    } else                                                  \
      return DEFAULT;                                       \
  }

#define PY_EXPR_VISITOR_DISPATCH(OP, PY_NAME, POST_ORDER_NAME)           \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) { \
    auto post_it = self->operator->()->map_.find(POST_ORDER_NAME);       \
    if (post_it != self->operator->()->map_.end()) {                     \
      self->VisitExprPostOrder_(static_cast<const OP*>(n.get()));        \
      post_it->second(n);                                                \
    } else {                                                             \
      auto it = self->operator->()->map_.find(PY_NAME);                  \
      if (it != self->operator->()->map_.end())                          \
        it->second(n);                                                   \
      else                                                               \
        self->VisitExpr_(static_cast<const OP*>(n.get()));               \
    }                                                                    \
  });

#define PY_EXPR_MUTATOR_DISPATCH(OP, PY_NAME, POST_ORDER_NAME)                \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) {      \
    auto post_it = self->operator->()->map_.find(POST_ORDER_NAME);            \
    if (post_it != self->operator->()->map_.end()) {                          \
      Expr expr = self->VisitExprPostOrder_(static_cast<const OP*>(n.get())); \
      expr = post_it->second(expr);                                           \
      return expr;                                                            \
    } else {                                                                  \
      auto it = self->operator->()->map_.find(PY_NAME);                       \
      if (it != self->operator->()->map_.end()) {                             \
        Expr expr = it->second(n);                                            \
        return expr;                                                          \
      } else                                                                  \
        return self->VisitExpr_(static_cast<const OP*>(n.get()));             \
    }                                                                         \
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
  virtual R VisitExpr_(const RuntimeDepShapeNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
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
    RELAX_EXPR_FUNCTOR_DISPATCH(RuntimeDepShapeNode);
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

/*!
 * \brief A simple visitor wrapper around ExprFunctor.
 *  Recursively visit the content.
 */
class ExprVisitor : public ExprFunctor<void(const Expr&)> {
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
  void VisitExpr_(const RuntimeDepShapeNode* op) override;
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
  virtual void VisitBinding(const Binding& binding);
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingNode* binding);
  virtual void VisitBinding_(const MatchShapeNode* binding);

  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   */
  virtual void VisitBindingBlock(const BindingBlock& block);
  // specific leaf level visitor functions
  virtual void VisitBindingBlock_(const BindingBlockNode* block);
  virtual void VisitBindingBlock_(const DataflowBlockNode* block);

  /*!
   * \brief Generic dispatcher for visiting the var definition site.
   * \param var The var to be visited.
   * \note VisitExpr_(const VarNode*) will only visit the usage site of an Var
   */
  virtual void VisitVarDef(const Var& var);
  // specific leaf level visitor functions
  virtual void VisitVarDef_(const VarNode* var);
  virtual void VisitVarDef_(const DataflowVarNode* var);

  virtual void VisitType(const Type& t);
  virtual void VisitSpan(const Span& span);
};

void PostOrderVisit(const Expr& node, std::function<void(const Expr&)> fvisit);

/*!
 * \brief A mutator works in unnormalized form.
 *
 * ExprMutatorBase expects input AST to be in the unnormalized form, i.e., checked_type_ and shape_
 * of expressions can be nullptr, and the expressions may nest(and as a result the AST is not in
 * ANF).
 */

class ExprMutatorBase : public ExprFunctor<Expr(const Expr&)> {
 public:
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const DataflowVarNode* op) override;
  Expr VisitExpr_(const ShapeExprNode* op) override;
  Expr VisitExpr_(const RuntimeDepShapeNode* op) override;
  Expr VisitExpr_(const ExternFuncNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* op) override;
  Expr VisitExpr_(const SeqExprNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;

  /*!
   * \brief Mutate BindingBlock.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  virtual BindingBlock VisitBindingBlock(const BindingBlock& block);

  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
};

/*!
 * \brief A mutator works in normal form.
 *
 * ExprMutator expects input AST to be in the normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all checked_type_ and shape_ of expressions are
 * available.
 */
class ExprMutator : public ExprMutatorBase {
 public:
  using ExprMutatorBase::VisitExpr_;

  ExprMutator(Optional<IRModule> mod = NullOpt) { builder_ = BlockBuilder::Create(mod); }
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const DataflowVarNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const SeqExprNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  virtual void VisitBinding(const Binding& binding);
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingNode* binding);
  virtual void VisitBinding_(const MatchShapeNode* binding);

  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  virtual BindingBlock VisitBindingBlock(const BindingBlock& block) override;  // NOLINT(*)
  // specific leaf level visitor functions
  virtual BindingBlock VisitBindingBlock_(const BindingBlockNode* block);
  virtual BindingBlock VisitBindingBlock_(const DataflowBlockNode* block);

  /*!
   * \brief Generic dispatcher for rewriting the var definition site.
   * \param var The var to be visited.
   * \return The var after post-order rewritten.
   * \note VisitExpr_(const VarNode*) will only visit the usage site of an Var
   */
  virtual Var VisitVarDef(const Var& var);
  // specific leaf level visitor functions
  virtual Var VisitVarDef_(const VarNode* var);
  virtual Var VisitVarDef_(const DataflowVarNode* var);

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
   * \note For function parameters, this function returns NullOpt.
   */
  Optional<Expr> LookupBinding(const Var& var);

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
   * \brief Create a new var with specified shape and type if the original var's shape or type does
   * not match with the specified ones.
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

class PyExprVisitorNode : public Object {
 public:
  /*! TODO */
  std::unordered_map<std::string, PackedFunc> map_;

  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "expr_functor.PyExprVisitor";
  TVM_DECLARE_BASE_OBJECT_INFO(PyExprVisitorNode, Object);
};

TVM_REGISTER_NODE_TYPE(PyExprVisitorNode);

class PyExprVisitor : public ObjectRef, public ExprVisitor {
 private:
  using TSelf = PyExprVisitor;
  using FType = tvm::NodeFunctor<void(const ObjectRef& n, TSelf* self)>;

 public:
  TVM_DLL PyExprVisitor(std::unordered_map<std::string, PackedFunc> map) {
    ObjectPtr<PyExprVisitorNode> n = make_object<PyExprVisitorNode>();
    n->map_ = std::move(map);
    data_ = std::move(n);
  }

  void VisitExpr(const Expr& expr) {
    auto it = operator->()->map_.find("visit_expr");
    if (it != operator->()->map_.end())
      it->second(expr);
    else {
      static FType vtable = InitVTable();
      vtable(expr, this);
    }
  }

  void VisitBinding(const Binding& binding)
      PY_EXPR_VISITOR_DEFAULT(binding, "visit_binding", ExprVisitor::VisitBinding(binding));

  void VisitBinding_(const VarBindingNode* binding)
      PY_EXPR_VISITOR_DEFAULT(GetRef<VarBinding>(binding), "visit_var_binding_",
                              ExprVisitor::VisitBinding_(binding));
  void VisitBinding_(const MatchShapeNode* binding)
      PY_EXPR_VISITOR_DEFAULT(GetRef<MatchShape>(binding), "visit_match_shape_",
                              ExprVisitor::VisitBinding_(binding));

  void VisitBindingBlock(const BindingBlock& block)
      PY_EXPR_VISITOR_DEFAULT(block, "visit_binding_block", ExprVisitor::VisitBindingBlock(block));

  void VisitBindingBlock_(const BindingBlockNode* block)
      PY_EXPR_VISITOR_DEFAULT(GetRef<BindingBlock>(block), "visit_binding_block_",
                              ExprVisitor::VisitBindingBlock_(block));
  void VisitBindingBlock_(const DataflowBlockNode* block)
      PY_EXPR_VISITOR_DEFAULT(GetRef<DataflowBlock>(block), "visit_dataflow_block_",
                              ExprVisitor::VisitBindingBlock_(block));

  void VisitVarDef(const Var& var)
      PY_EXPR_VISITOR_DEFAULT(var, "visit_var_def", ExprVisitor::VisitVarDef(var));
  void VisitVarDef_(const VarNode* var)
      PY_EXPR_VISITOR_DEFAULT(GetRef<Var>(var), "visit_var_def_", ExprVisitor::VisitVarDef_(var));
  void VisitVarDef_(const DataflowVarNode* var)
      PY_EXPR_VISITOR_DEFAULT(GetRef<DataflowVar>(var), "visit_dataflow_var_def_",
                              ExprVisitor::VisitVarDef_(var));

  void VisitType(const Type& t) PY_EXPR_VISITOR_DEFAULT(t, "visit_type", ExprVisitor::VisitType(t));
  void VisitSpan(const Span& span)
      PY_EXPR_VISITOR_DEFAULT(span, "visit_span", ExprVisitor::VisitSpan(span));

  template <typename T>
  void VisitExprPostOrder_(const T* op) {
    ExprVisitor::VisitExpr_(op);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(PyExprVisitor, ObjectRef, PyExprVisitorNode);

 private:
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    PY_EXPR_VISITOR_DISPATCH(ConstantNode, "visit_constant_", "rewrite_constant_post_order");
    PY_EXPR_VISITOR_DISPATCH(TupleNode, "visit_tuple_", "rewrite_tuple_post_order");
    PY_EXPR_VISITOR_DISPATCH(VarNode, "visit_var_", "rewrite_var_post_order");
    PY_EXPR_VISITOR_DISPATCH(DataflowVarNode, "visit_dataflow_var_",
                             "rewrite_dataflow_var_post_order");
    PY_EXPR_VISITOR_DISPATCH(ShapeExprNode, "visit_shape_expr_", "rewrite_shape_expr_post_order");
    PY_EXPR_VISITOR_DISPATCH(RuntimeDepShapeNode, "visit_runtime_dep_shape_",
                             "rewrite_runtime_dep_post_order");
    PY_EXPR_VISITOR_DISPATCH(ExternFuncNode, "visit_extern_func_",
                             "rewrite_extern_func_post_order");
    PY_EXPR_VISITOR_DISPATCH(GlobalVarNode, "visit_global_var_", "rewrite_global_var_post_order");
    PY_EXPR_VISITOR_DISPATCH(FunctionNode, "visit_function_", "rewrite_function_post_order");
    PY_EXPR_VISITOR_DISPATCH(CallNode, "visit_call_", "rewrite_call_post_order");
    PY_EXPR_VISITOR_DISPATCH(SeqExprNode, "visit_seq_expr_", "rewrite_seq_expr_post_order");
    PY_EXPR_VISITOR_DISPATCH(IfNode, "visit_if_", "rewrite_if_post_order");
    PY_EXPR_VISITOR_DISPATCH(OpNode, "visit_op_", "rewrite_op_post_order");
    PY_EXPR_VISITOR_DISPATCH(TupleGetItemNode, "visit_tuple_getitem_",
                             "rewrite_tuple_getitem_post_order");
    return vtable;
  }
};

class PyExprMutatorNode : public Object {
 public:
  /*! TODO */
  std::unordered_map<std::string, PackedFunc> map_;

  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "expr_functor.PyExprMutator";
  TVM_DECLARE_BASE_OBJECT_INFO(PyExprMutatorNode, Object);
};

TVM_REGISTER_NODE_TYPE(PyExprMutatorNode);

class PyExprMutator : public ObjectRef, public ExprMutator {
 private:
  using TSelf = PyExprMutator;
  using FType = tvm::NodeFunctor<Expr(const ObjectRef& n, TSelf* self)>;

 public:
  TVM_DLL PyExprMutator(std::unordered_map<std::string, PackedFunc> map) {
    ObjectPtr<PyExprMutatorNode> n = make_object<PyExprMutatorNode>();
    n->map_ = std::move(map);
    data_ = std::move(n);
  }

  Expr VisitExpr(const Expr& expr) {
    auto it = operator->()->map_.find("visit_expr");
    if (it != operator->()->map_.end())
      return builder_->Normalize(it->second(expr));
    else {
      static FType vtable = InitVTable();
      return builder_->Normalize(vtable(expr, this));
    }
  }

  void VisitBinding(const Binding& binding)
      PY_EXPR_VISITOR_DEFAULT(binding, "visit_binding", ExprMutator::VisitBinding(binding));

  void VisitBinding_(const VarBindingNode* binding)
      PY_EXPR_VISITOR_DEFAULT(GetRef<VarBinding>(binding), "visit_var_binding_",
                              ExprMutator::VisitBinding_(binding));
  void VisitBinding_(const MatchShapeNode* binding)
      PY_EXPR_VISITOR_DEFAULT(GetRef<MatchShape>(binding), "visit_match_shape_",
                              ExprMutator::VisitBinding_(binding));

  BindingBlock VisitBindingBlock(const BindingBlock& block)
      PY_EXPR_MUTATOR_DEFAULT(block, "visit_binding_block", ExprMutator::VisitBindingBlock(block),
                              BindingBlock);

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block)
      PY_EXPR_MUTATOR_DEFAULT(GetRef<BindingBlock>(block), "visit_binding_block_",
                              ExprMutator::VisitBindingBlock_(block), BindingBlock);
  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block)
      PY_EXPR_MUTATOR_DEFAULT(GetRef<DataflowBlock>(block), "visit_dataflow_block_",
                              ExprMutator::VisitBindingBlock_(block), BindingBlock);

  Var VisitVarDef(const Var& var)
      PY_EXPR_MUTATOR_DEFAULT(var, "visit_var_def", ExprMutator::VisitVarDef(var), Var);
  Var VisitVarDef_(const VarNode* var) PY_EXPR_MUTATOR_DEFAULT(GetRef<Var>(var), "visit_var_def_",
                                                               ExprMutator::VisitVarDef_(var), Var);
  Var VisitVarDef_(const DataflowVarNode* var)
      PY_EXPR_MUTATOR_DEFAULT(GetRef<DataflowVar>(var), "visit_dataflow_var_def_",
                              ExprMutator::VisitVarDef_(var), Var);

  Type VisitType(const Type& t)
      PY_EXPR_MUTATOR_DEFAULT(t, "visit_type", ExprMutator::VisitType(t), Type);

  TVM_DEFINE_OBJECT_REF_METHODS(PyExprMutator, ObjectRef, PyExprMutatorNode);

 private:
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    PY_EXPR_MUTATOR_DISPATCH(ConstantNode, "visit_constant_", "rewrite_constant_post_order");
    PY_EXPR_MUTATOR_DISPATCH(TupleNode, "visit_tuple_", "rewrite_tuple_post_order");
    PY_EXPR_MUTATOR_DISPATCH(VarNode, "visit_var_", "rewrite_var_post_order");
    PY_EXPR_MUTATOR_DISPATCH(DataflowVarNode, "visit_dataflow_var_",
                             "rewrite_dataflow_var_post_order");
    PY_EXPR_MUTATOR_DISPATCH(ShapeExprNode, "visit_shape_expr_", "rewrite_shape_expr_post_order");
    PY_EXPR_MUTATOR_DISPATCH(RuntimeDepShapeNode, "visit_runtime_dep_shape_",
                             "rewrite_runtime_dep_post_order");
    PY_EXPR_MUTATOR_DISPATCH(ExternFuncNode, "visit_extern_func_",
                             "rewrite_extern_func_post_order");
    PY_EXPR_MUTATOR_DISPATCH(GlobalVarNode, "visit_global_var_", "rewrite_global_var_post_order");
    PY_EXPR_MUTATOR_DISPATCH(FunctionNode, "visit_function_", "rewrite_function_post_order");
    PY_EXPR_MUTATOR_DISPATCH(CallNode, "visit_call_", "rewrite_call_post_order");
    PY_EXPR_MUTATOR_DISPATCH(SeqExprNode, "visit_seq_expr_", "rewrite_seq_expr_post_order");
    PY_EXPR_MUTATOR_DISPATCH(IfNode, "visit_if_", "rewrite_if_post_order");
    PY_EXPR_MUTATOR_DISPATCH(OpNode, "visit_op_", "rewrite_op_post_order");
    PY_EXPR_MUTATOR_DISPATCH(TupleGetItemNode, "visit_tuple_getitem_",
                             "rewrite_tuple_getitem_post_order");
    return vtable;
  }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_EXPR_FUNCTOR_H_
