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
 * \file tvm/relax/ir_functor.h
 * \brief A generic functor for working with Relax IR nodes.
 * \sa tvm/relax/expr_functor.h for common IR rewriting use-cases.
 */
#ifndef TVM_RELAX_IR_FUNCTOR_H_
#define TVM_RELAX_IR_FUNCTOR_H_

#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/expr.h>

#include <utility>

namespace tvm {
namespace relax {

template <typename FType>
class IRFunctor;

#define IR_FUNCTOR_DEFAULT \
  { return VisitNodeDefault_(op, std::forward<Args>(args)...); }

#define RELAX_IR_FUNCTOR_DISPATCH(OP)                                                      \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitNode_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class IRFunctor<R(const ObjectRef& n, Args...)> {
 private:
  using TSelf = IRFunctor<R(const ObjectRef& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  using result_type = R;
  virtual ~IRFunctor() {}

  R operator()(const ObjectRef& n, Args... args) {
    return VisitNode(n, std::forward<Args>(args)...);
  }

  virtual R VisitNode(const ObjectRef& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }

  // IR nodes inherited from Relay
  virtual R VisitNode_(const relay::ConstantNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relay::TupleNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relay::GlobalVarNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relay::CallNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relay::IfNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const OpNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relay::TupleGetItemNode* op, Args... args) IR_FUNCTOR_DEFAULT;

  // IR nodes introduced by Relax
  virtual R VisitNode_(const relax::VarNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::DataflowVarNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::ShapeExprNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::RuntimeDepShapeNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::MatchShapeNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::VarBindingNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::BindingBlockNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::DataflowBlockNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::SeqExprNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::FunctionNode* op, Args... args) IR_FUNCTOR_DEFAULT;
  virtual R VisitNode_(const relax::ExternFuncNode* op, Args... args) IR_FUNCTOR_DEFAULT;

  virtual R VisitNodeDefault_(const Object* op, Args...) {
    LOG(FATAL) << "no default visitor implemented for " << op->GetTypeKey();
    throw;
  }

 private:
  static FType InitVTable() {
    FType vtable;
    RELAX_IR_FUNCTOR_DISPATCH(relay::ConstantNode);
    RELAX_IR_FUNCTOR_DISPATCH(relay::TupleNode);
    RELAX_IR_FUNCTOR_DISPATCH(relay::GlobalVarNode);
    RELAX_IR_FUNCTOR_DISPATCH(relay::CallNode);
    RELAX_IR_FUNCTOR_DISPATCH(relay::IfNode);
    RELAX_IR_FUNCTOR_DISPATCH(OpNode);
    RELAX_IR_FUNCTOR_DISPATCH(relay::TupleGetItemNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::VarNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::DataflowVarNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::ShapeExprNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::RuntimeDepShapeNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::MatchShapeNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::VarBindingNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::BindingBlockNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::DataflowBlockNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::SeqExprNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::FunctionNode);
    RELAX_IR_FUNCTOR_DISPATCH(relax::ExternFuncNode);
    return vtable;
  }
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_FUNCTOR_H_
