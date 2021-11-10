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

void ExprVisitor::VisitVarBinding(const VarBinding& binding) {
  this->VisitExpr(binding->value);
  this->VisitExpr(binding->var);
}

void ExprVisitor::VisitMatchShape(const MatchShape& binding) {
  this->VisitExpr(binding->value);
  // TODO(ziheng): should we change pattern from
  // Array<PrimExpr> to ShapeExpr?
  this->VisitExpr(ShapeExpr(binding->pattern));
  if (binding->var.defined()) {
    this->VisitExpr(binding->var);
  }
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

Expr ExprMutator::VisitExpr_(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const TupleNode* op) {
  tvm::Array<Expr> fields;
  bool all_fields_unchanged = true;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    all_fields_unchanged &= new_field.same_as(field);
  }

  if (all_fields_unchanged) {
    return GetRef<Expr>(op);
  } else {
    return Tuple(fields, op->span);
  }
}

// Visit the use-site of a defined Var
Expr ExprMutator::VisitExpr_(const VarNode* op) {
  auto it = var_remap_.find(op->vid);
  if (it != var_remap_.end()) {
    return it->second;
  }

  // default case return self.
  return GetRef<Expr>(op);
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutator::VisitExpr_(const DataflowVarNode* op) {
  auto it = var_remap_.find(op->vid);
  if (it != var_remap_.end()) {
    return it->second;
  }

  // default case return self.
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<Var> params;
  bool all_params_unchanged = true;
  for (Var param : op->params) {
    Var new_param = this->VisitVarDef(param);
    params.push_back(new_param);
    all_params_unchanged &= param.same_as(new_param);
  }

  Type ret_type = this->VisitType(op->ret_type);
  Expr body = this->VisitWithNewScope(op->body);

  if (all_params_unchanged && ret_type.same_as(op->ret_type) && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(op->name, params, body, ret_type);
  }
}

Expr ExprMutator::VisitExpr_(const CallNode* call_node) {
  Expr new_op = this->VisitExpr(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  tvm::Array<Type> ty_args;
  for (Type ty_arg : call_node->type_args) {
    Type new_ty_arg = this->VisitType(ty_arg);
    ty_args.push_back(new_ty_arg);
    unchanged &= new_ty_arg.same_as(ty_arg);
  }

  tvm::Array<Expr> call_args;
  for (Expr arg : call_node->args) {
    Expr new_arg = this->VisitExpr(arg);
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
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitWithNewScope(op->true_branch);
  Expr false_b = this->VisitWithNewScope(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutator::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const TupleGetItemNode* get_item) {
  auto t = this->VisitExpr(get_item->tuple);
  if (get_item->tuple.same_as(t)) {
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
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  builder_->BeginBindingBlock();
  Expr body = this->VisitExpr(op->body);
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

void ExprMutator::VisitVarBinding(const VarBinding& binding) {
  Expr new_value = this->VisitExpr(binding->value);
  Var new_var = this->VisitVarDef(binding->var);

  if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
    // no-op if there is no change
    builder_->Emit(binding);
    return;
  }

  {
    Var temp = WithShapeAndType(new_var, new_value->shape_, new_value->checked_type_);
    if (!temp.same_as(new_var)) {
      new_var = temp;
      this->var_remap_[binding->var->vid] = new_var;
    }
  }

  if (builder_->CurrentBlockIsDataFlow() && !new_var.as<DataflowVarNode>()) {
    builder_->EmitOutput(VarBinding(new_var, new_value));
  } else {
    builder_->Emit(VarBinding(new_var, new_value));
  }
}

void ExprMutator::VisitMatchShape(const MatchShape& binding) {
  Expr new_value = this->VisitExpr(binding->value);
  Expr new_pattern = this->VisitExpr(ShapeExpr(binding->pattern));

  Var new_var;
  if (binding->var.defined()) {
    // in the case of `x = R.match_shape(val, pattern)`, we want `x` to directly get `pattern` as
    // the shape when `val` is a tensor.
    Optional<Expr> new_shape;
    if (new_value->checked_type_.defined() && new_value->checked_type_.as<DynTensorTypeNode>()) {
      new_shape = new_pattern;
    }
    Var temp =
        WithShapeAndType(this->VisitVarDef(binding->var), new_shape, new_value->checked_type_);
    if (!temp.same_as(new_var)) {
      new_var = temp;
      this->var_remap_[binding->var->vid] = new_var;
    }
  }

  // TODO(@altanh, @yuchen): shape and type inference here too...
  // TODO: when value's shape/type changed, create new var
  // TODO: group the can prove shape/type logic and replace var into a function
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

Var ExprMutator::VisitVarDef(const Var& var) {
  if (var->type_annotation.defined()) {
    Type type = this->VisitType(var->type_annotation.value());
    if (!var->type_annotation.same_as(type)) {
      Var new_var;
      if (var.as<DataflowVarNode>()) {
        new_var = DataflowVar(var->vid, NullOpt, type, var->span);
      } else {
        new_var = Var(var->vid, NullOpt, type, var->span);
      }
      new_var->shape_ = var->shape_;
      this->var_remap_[var->vid] = new_var;
      return new_var;
    }
  }
  return var;
}

Expr ExprMutator::VisitExpr(const Expr& expr) {
  return builder_->Normalize(ExprFunctor::VisitExpr(expr));
}

Expr ExprMutator::VisitWithNewScope(const Expr& expr) {
  builder_->BeginBindingBlock();
  Expr ret = this->VisitExpr(expr);
  BindingBlock prologue = builder_->EndBlock();
  if (!prologue->bindings.empty()) {
    ret = SeqExpr({prologue}, ret);
  }
  return ret;
}

Expr ExprMutator::LookupBinding(const Var& var) { return builder_->LookupBinding(var); }

Var ExprMutator::WithShapeAndType(Var var, Optional<ObjectRef> shape, Type type) {
  // shape/type changes if it goes from defined -> undefined or the other way, hence xor
  bool shape_changed = var->shape_.operator bool() ^ shape.operator bool();
  shape_changed |= var->shape_ && shape &&
                   !builder_->CanProveShapeEqual(Downcast<Expr>(var->shape_.value()),
                                                 Downcast<Expr>(shape.value()));

  bool type_changed = var->checked_type_.defined() ^ type.defined();
  type_changed |= var->checked_type_.defined() && type.defined() &&
                  !StructuralEqual()(var->checked_type_, type);

  if (shape_changed || type_changed) {
    Var new_var = var.as<DataflowVarNode>() ? DataflowVar(var->vid, NullOpt, NullOpt, var->span)
                                            : Var(var->vid, NullOpt, NullOpt, var->span);
    new_var->shape_ = var->shape_;
    new_var->checked_type_ = var->checked_type_;
    var = new_var;
  }

  if (shape_changed) {
    var->shape_ = shape;
  }

  if (type_changed) {
    var->checked_type_ = type;
  }

  return var;
}

// ==================
// DataflowMutator

void DataflowMutator::VisitBinding(const Binding& binding) {
  if (binding.as<VarBindingNode>()) {
    VarBinding var_binding = Downcast<VarBinding>(binding);
    if (builder_->CurrentBlockIsDataFlow()) {
      this->VisitDataflowVarBinding(var_binding);
    } else {
      ExprMutator::VisitVarBinding(var_binding);
    }
  } else {
    ExprMutator::VisitBinding(binding);
  }
}

void DataflowMutator::VisitDataflowVarBinding(const VarBinding& binding) {
  ExprMutator::VisitVarBinding(binding);
}

}  // namespace relax
}  // namespace tvm
