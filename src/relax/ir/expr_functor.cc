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
 * \file src/relax/expr_functor.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

void ExprVisitor::VisitExpr_(const ConstantNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (auto field : op->fields) {
    this->VisitExpr(field);
  }
}

void ExprVisitor::VisitExpr_(const VarNode* op) {
  this->VisitSpan(op->span);
  if (op->type_annotation.defined()) {
    this->VisitType(op->type_annotation.value());
  }
}

void ExprVisitor::VisitExpr_(const DataflowVarNode* op) {
  this->VisitSpan(op->span);
  if (op->type_annotation.defined()) {
    this->VisitType(op->type_annotation.value());
  }
}

void ExprVisitor::VisitExpr_(const FunctionNode* op) {
  this->VisitSpan(op->span);
  for (auto param : op->params) {
    this->VisitExpr(param);
  }

  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (auto ty_arg : op->type_args) {
    this->VisitType(ty_arg);
  }

  for (auto arg : op->args) {
    this->VisitExpr(arg);
  }
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}

void ExprVisitor::VisitExpr_(const OpNode* op) {
  return;
}

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->tuple);
}

void ExprVisitor::VisitExpr_(const ShapeExprNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const ExternFuncNode* op) {
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const SeqExprNode* op) {
  this->VisitSpan(op->span);
  for (auto block : op->blocks) {
    this->VisitBindingBlock(block);
  }
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitType(const Type& t) {
  return;
}

void ExprVisitor::VisitSpan(const Span& span) {
  return;
}

void ExprVisitor::VisitBinding(const Binding& binding) {
  if (binding.as<VarBindingNode>()) {
    this->VisitVarBinding(Downcast<VarBinding>(binding));
  } else if (binding.as<MatchShapeNode>()) {
    this->VisitMatchShape(Downcast<MatchShape>(binding));
  } else {
    LOG(FATAL) << "Wrong type.";
  }
}

void ExprVisitor::VisitVarBinding(const VarBinding& binding) {
  this->VisitExpr(binding->value);
}

void ExprVisitor::VisitMatchShape(const MatchShape& binding) {
  this->VisitExpr(binding->value);
  // TODO(ziheng): should we change pattern from
  // Array<PrimExpr> to ShapeExpr?
  this->VisitExpr(ShapeExpr(binding->pattern));
}

void ExprVisitor::VisitBindingBlock(const BindingBlock& block) {
  if (block.as<DataflowBlockNode>()) {
    this->VisitDataflowBlock(Downcast<DataflowBlock>(block));
  } else {
    for (auto binding : block->bindings) {
      this->VisitBinding(binding);
    }
  }
}

void ExprVisitor::VisitDataflowBlock(const DataflowBlock& block) {
  for (auto binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitExpr(const Expr& expr) {
  using TParent = ExprFunctor<void(const Expr&)>;
  TParent::VisitExpr(expr);
}

class ExprApplyVisit : public ExprVisitor {
 public:
  explicit ExprApplyVisit(std::function<void(const Expr&)> f) : f_(f) {}

  void VisitExpr(const Expr& e) final {
    ExprVisitor::VisitExpr(e);
    f_(e);
  }

 private:
  std::function<void(const Expr&)> f_;
};

void PostOrderVisit(const Expr& e, std::function<void(const Expr&)> fvisit) {
  ExprApplyVisit(fvisit).VisitExpr(e);
}

TVM_REGISTER_GLOBAL("relax.analysis.post_order_visit")
.set_body_typed([](Expr expr, PackedFunc f) {
  PostOrderVisit(expr, [f](const Expr& n) { f(n); });
});


// ==================
// ExprMutator

Expr ExprMutator::VisitExpr_(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const TupleNode* op) {
  tvm::Array<Expr> fields;
  bool all_fields_unchanged = true;
  for (auto field : op->fields) {
    auto new_field = this->Mutate(field);
    fields.push_back(new_field);
    all_fields_unchanged &= new_field.same_as(field);
  }

  if (all_fields_unchanged) {
    return GetRef<Expr>(op);
  } else {
    return Tuple(fields, op->span);
  }
}

Expr ExprMutator::VisitExpr_(const VarNode* op) {
  if (op->type_annotation.defined()) {
    auto type = this->VisitType(op->type_annotation.value());
    if (!op->type_annotation.same_as(type)) {
      return Var(op->vid, Downcast<Expr>(op->shape()), type, op->span);
    }
  }
  // default case return self.
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const DataflowVarNode* op) {
  if (op->type_annotation.defined()) {
    auto type = this->VisitType(op->type_annotation.value());
    if (!op->type_annotation.same_as(type)) {
      return DataflowVar(op->vid, Downcast<Expr>(op->shape()), type, op->span);
    }
  }
  // default case return self.
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<Var> params;
  bool all_params_unchanged = true;
  for (auto param : op->params) {
    Var new_param = Downcast<Var>(this->Mutate(param));
    params.push_back(new_param);
    all_params_unchanged &= param.same_as(new_param);
  }

  auto ret_type = this->VisitType(op->ret_type);
  auto body = this->Mutate(op->body);

  if (all_params_unchanged && ret_type.same_as(op->ret_type) && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(op->name, params, body, ret_type);
  }
}

Expr ExprMutator::VisitExpr_(const CallNode* call_node) {
  auto new_op = this->Mutate(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  tvm::Array<Type> ty_args;
  for (auto ty_arg : call_node->type_args) {
    auto new_ty_arg = this->VisitType(ty_arg);
    ty_args.push_back(new_ty_arg);
    unchanged &= new_ty_arg.same_as(ty_arg);
  }

  tvm::Array<Expr> call_args;
  for (auto arg : call_node->args) {
    auto new_arg = this->Mutate(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  if (unchanged) {
    return GetRef<Expr>(call_node);
  } else {
    return Call(new_op, call_args, call_node->attrs, ty_args, call_node->span);
  }
}

Expr ExprMutator::VisitExpr_(const IfNode* op) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutator::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const TupleGetItemNode* get_item) {
  auto t = this->Mutate(get_item->tuple);
  if (get_item->tuple == t) {
    return GetRef<Expr>(get_item);
  } else {
    return TupleGetItem(t, get_item->index, get_item->span);
  }
}

Expr ExprMutator::VisitExpr_(const ShapeExprNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const ExternFuncNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    blocks.push_back(new_block);
    all_blocks_unchanged &= block.same_as(new_block);
  }

  Expr body = this->Mutate(op->body);
  if (all_blocks_unchanged && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return SeqExpr(blocks, body);
  }
}

Type ExprMutator::VisitType(const Type& t) { return t; }

void ExprMutator::VisitBinding(const Binding& binding, IRBuilder& builder) {
  Binding new_binding;
  if (binding.as<VarBindingNode>()) {
    this->VisitVarBinding(Downcast<VarBinding>(binding), builder);
  } else if (binding.as<MatchShapeNode>()) {
    this->VisitMatchShape(Downcast<MatchShape>(binding), builder);
  } else {
    LOG(FATAL) << "Wrong type.";
  }
}

Var ExprMutator::VisitVarBinding(const VarBinding& binding, IRBuilder& builder) {
  Expr new_value = this->Mutate(binding->value);
  if (!binding->var.as<DataflowVarNode>()) {
    return builder->EmitOutput(new_value);
  } else {
    return builder->Emit(Downcast<Call>(new_value));
  }
}

void ExprMutator::VisitMatchShape(const MatchShape& binding, IRBuilder& builder) {
  this->Mutate(binding->value);
  this->Mutate(ShapeExpr(binding->pattern));
}

BindingBlock ExprMutator::VisitBindingBlock(const BindingBlock& block) {
  if (block.as<DataflowBlockNode>()) {
    return this->VisitDataflowBlock(Downcast<DataflowBlock>(block));
  } else{
    this->builder_ = IRBuilderNode::Create();
    for (auto binding : block->bindings) {
      this->VisitBinding(binding, this->builder_);
    }
    auto blocks = this->builder_->GetBlocks();
    return blocks.back();
  }
}

BindingBlock ExprMutator::VisitDataflowBlock(const DataflowBlock& block) {
  this->builder_ = LazyIRBuilderNode::Create(block);
  {
    With<DataflowScope> scope(this->builder_);
    for (auto binding : block->bindings) {
      this->VisitBinding(binding, this->builder_);
    }
  }
  return this->builder_->GetBlocks().back();
}

Expr ExprMutator::VisitExpr(const Expr& expr) {
  Expr new_expr = ExprFunctor::VisitExpr(expr);
  return new_expr;
}


// ==================
// DataflowMutator

BindingBlock DataflowMutator::VisitDataflowBlock(const DataflowBlock& block) {
  this->builder_ = LazyIRBuilderNode::Create(block);
  {
    With<DataflowScope> scope(this->builder_);
    for (auto binding : block->bindings) {
      if (auto* var_binding = binding.as<VarBindingNode>()) {
        Var var = this->VisitVarBinding(Downcast<VarBinding>(binding), this->builder_);
        this->pre_post_var_map_[var_binding->var] = var;
      }
    }
  }
  return this->builder_->GetBlocks().back();
}

Var DataflowMutator::VisitVarBinding(const VarBinding& binding, IRBuilder& builder) {
  Expr new_value = this->Mutate(binding->value);
  Var new_var;
  if (new_value.as<CallNode>()) {
    new_var = builder->Emit(Downcast<Call>(new_value));
  }
  if (!binding->var.as<DataflowVarNode>()) {
    new_var = builder->EmitOutput(new_value);
  }
  pre_post_var_map_[binding->var] = new_var;
  return new_var;
}

Expr DataflowMutator::LookupVar(Var var) {
  auto it = pre_post_var_map_.find(var);
  if (it != pre_post_var_map_.end()) {
    return builder_->LookupVar(it->first);
  } else {
    return builder_->LookupVar(var);
  }
}
}  // namespace relax
}  // namespace tvm
