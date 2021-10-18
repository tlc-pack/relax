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
#include <tvm/relax/type.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/pattern_functor.h>

namespace tvm {
namespace relax {

void ExprVisitor::VisitExpr_(const ConstantNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (Expr field : op->fields) {
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
  for (Var param : op->params) {
    this->VisitExpr(param);
  }

  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (Type ty_arg : op->type_args) {
    this->VisitType(ty_arg);
  }

  for (Expr arg : op->args) {
    this->VisitExpr(arg);
  }
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}

void ExprVisitor::VisitExpr_(const OpNode* op) {}

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->tuple);
}

void ExprVisitor::VisitExpr_(const ShapeExprNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const ExternFuncNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const SeqExprNode* op) {
  this->VisitSpan(op->span);
  for (BindingBlock block : op->blocks) {
    this->VisitBindingBlock(block);
  }
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitType(const Type& t) {}

void ExprVisitor::VisitSpan(const Span& span) {}

void ExprVisitor::VisitBinding(const Binding& binding) {
  if (binding.as<VarBindingNode>()) {
    this->VisitVarBinding(Downcast<VarBinding>(binding));
  } else if (binding.as<MatchShapeNode>()) {
    this->VisitMatchShape(Downcast<MatchShape>(binding));
  } else {
    LOG(FATAL) << "Wrong type.";
  }
}

void ExprVisitor::VisitVarBinding(const VarBinding& binding) { this->VisitExpr(binding->value); }

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
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
  }
}

void ExprVisitor::VisitDataflowBlock(const DataflowBlock& block) {
  for (Binding binding : block->bindings) {
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

TVM_REGISTER_GLOBAL("relax.analysis.post_order_visit").set_body_typed([](Expr expr, PackedFunc f) {
  PostOrderVisit(expr, [f](const Expr& n) { f(n); });
});

// ==================
// ExprMutator

Expr ExprMutator::VisitExpr_(const ConstantNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const TupleNode* op) {
  tvm::Array<Expr> fields;
  bool all_fields_unchanged = true;
  for (Expr field : op->fields) {
    Expr new_field = this->Mutate(field);
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
  auto it = var_remap_.find(GetRef<Var>(op));
  if (it != var_remap_.end()) {
    return it->second;
  }
  if (op->type_annotation.defined()) {
    Type type = this->VisitType(op->type_annotation.value());
    if (!op->type_annotation.same_as(type)) {
      return Var(op->vid, Downcast<Expr>(op->shape()), type, op->span);
    }
  }
  // default case return self.
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const DataflowVarNode* op) {
  if (op->type_annotation.defined()) {
    Type type = this->VisitType(op->type_annotation.value());
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
  for (Var param : op->params) {
    Var new_param = Downcast<Var>(this->Mutate(param));
    params.push_back(new_param);
    all_params_unchanged &= param.same_as(new_param);
  }

  Type ret_type = this->VisitType(op->ret_type);
  Expr body = this->MutateWithPrologue(op->body, false);

  if (all_params_unchanged && ret_type.same_as(op->ret_type) && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(op->name, params, body, ret_type);
  }
}

Expr ExprMutator::VisitExpr_(const CallNode* call_node) {
  Expr new_op = this->Mutate(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  tvm::Array<Type> ty_args;
  for (Type ty_arg : call_node->type_args) {
    Type new_ty_arg = this->VisitType(ty_arg);
    ty_args.push_back(new_ty_arg);
    unchanged &= new_ty_arg.same_as(ty_arg);
  }

  tvm::Array<Expr> call_args;
  for (Expr arg : call_node->args) {
    Expr new_arg = this->Mutate(arg);
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
  Expr guard = this->Mutate(op->cond);
  Expr true_b = this->MutateWithPrologue(op->true_branch, false);
  Expr false_b = this->MutateWithPrologue(op->false_branch, false);
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

Expr ExprMutator::VisitExpr_(const ShapeExprNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const ExternFuncNode* op) {
  return GetRef<Expr>(op); 
}

Expr ExprMutator::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  builder_->BeginBindingBlock();
  Expr body = this->Mutate(op->body);
  BindingBlock prologue = builder_->EndBlock();
  if (!prologue->bindings.empty()) {
    blocks.push_back(prologue);
    all_blocks_unchanged = false;
  }

  if (all_blocks_unchanged && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return SeqExpr(blocks, body);
  }
}

Type ExprMutator::VisitType(const Type& t) { return t; }

void ExprMutator::VisitBinding(const Binding& binding) {
  if (binding.as<VarBindingNode>()) {
    this->VisitVarBinding(Downcast<VarBinding>(binding));
  } else if (binding.as<MatchShapeNode>()) {
    this->VisitMatchShape(Downcast<MatchShape>(binding));
  } else {
    LOG(FATAL) << "Wrong type.";
  }
}

Var ExprMutator::VisitVarBinding(const VarBinding& binding) {
  Expr new_value = builder_->Normalize(this->Mutate(binding->value));
  Var new_var = Downcast<Var>(this->Mutate(binding->var));
  // TODO(@altanh): this probably shouldn't live here, all passes would have to make sure to do it
  //                in this method...
  // if (new_value->shape_.defined()) {
  //   if (new_var->shape_.defined()) {
  //     new_var = Var(new_var->vid, NullOpt, new_var->type_annotation, new_var->span);
  //   }
  //   new_var->shape_ = new_value->shape_;
  // }
  // if (new_value->checked_type_.defined()) {
  //   if (new_var->checked_type_.defined()) {

  //   }
  //   new_var = Var(new_var->vid, new_var->shape_, NullOpt, new_var->span);
  //   new_var->checked_type_ = new_value->checked_type_;
  // }
  
  if (!builder_->CanProveShapeEqual(new_var->shape(), new_value->shape()) ||
      !StructuralEqual()(new_var->checked_type(), new_value->checked_type())) {
    new_var = Var(new_var->vid, NullOpt, NullOpt, new_var->span);
    if (new_value->shape_.defined()) {
      new_var->shape_ = new_value->shape_;
    }
    // TODO(@yuchen, @altanh): checked_type_.defined() needs to change depends on how to represent unknown type
    if (new_value->checked_type_.defined()){
      new_var->checked_type_ = new_value->checked_type_;
    }
  }

  this->var_remap_[binding->var] = new_var;

  if (builder_->CurrentBlockIsDataFlow() && !binding->var.as<DataflowVarNode>()) {
    return builder_->EmitOutput(VarBinding(new_var, new_value));
  } else {
    return builder_->Emit(VarBinding(new_var, new_value));
  }
}

void ExprMutator::VisitMatchShape(const MatchShape& binding) {
  Expr new_value = this->Mutate(binding->value);
  Expr new_pattern = this->Mutate(ShapeExpr(binding->pattern));
  Var new_var = Downcast<Var>(this->Mutate(binding->var));
  builder_->EmitMatchShape(
      MatchShape(new_value, Downcast<ShapeExpr>(new_pattern)->values, new_var));
}

BindingBlock ExprMutator::VisitBindingBlock(const BindingBlock& block) {
  if (block.as<DataflowBlockNode>()) {
    return this->VisitDataflowBlock(Downcast<DataflowBlock>(block));
  } else {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }
}

BindingBlock ExprMutator::VisitDataflowBlock(const DataflowBlock& block) {
  builder_->BeginDataflowBlock();
  for (auto binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

Expr ExprMutator::VisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

Expr ExprMutator::MutateWithPrologue(const Expr& expr, bool is_dataflow) {
  if (is_dataflow) {
    builder_->BeginDataflowBlock();
  } else {
    builder_->BeginBindingBlock();
  }

  Expr ret = this->Mutate(expr);
  BindingBlock prologue = builder_->EndBlock();
  if (!prologue->bindings.empty()) {
    ret = SeqExpr({prologue}, ret);
  }
  return ret;
}

Expr ExprMutator::LookupVar(Var var) {
  auto it = var_remap_.find(var);
  if (it != var_remap_.end()) {
    return builder_->LookupVar(it->first);
  } else {
    return builder_->LookupVar(var);
  }
}

// ==================
// DataflowMutator

void DataflowMutator::VisitBinding(const Binding& binding) {
  if (binding.as<VarBindingNode>()) {
    VarBinding var_binding = Downcast<VarBinding>(binding);
    if (builder_->CurrentBlockIsDataFlow()) {
      var_remap_[var_binding->var] = this->VisitDataflowVarBinding(var_binding);
    } else {
      var_remap_[var_binding->var] = ExprMutator::VisitVarBinding(var_binding);
    }
  } else {
    ExprMutator::VisitBinding(binding);
  }
}

Var DataflowMutator::VisitDataflowVarBinding(const VarBinding& binding) {
  return ExprMutator::VisitVarBinding(binding);
}

}  // namespace relax
}  // namespace tvm
