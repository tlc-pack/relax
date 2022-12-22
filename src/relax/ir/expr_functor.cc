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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>

// functions to be overriden.
#define RELAX_VISIT_BINDING_DISPATCH(OP)                                   \
  vtable.template set_dispatch<OP>(                                        \
      [](const ObjectRef& n, TSelf* self, const VarBindingNode* binding) { \
        self->VisitBinding_(binding, static_cast<const OP*>(n.get()));     \
      });

#define RELAX_VAR_BINDING_DISPATCH_IMPL(Type)                                        \
  Type::VisitBindingVTable Type::InitVisitBindingVTable() {                          \
    VisitBindingVTable vtable;                                                       \
    RELAX_VISIT_BINDING_DISPATCH(ConstantNode);                                      \
    RELAX_VISIT_BINDING_DISPATCH(TupleNode);                                         \
    RELAX_VISIT_BINDING_DISPATCH(VarNode);                                           \
    RELAX_VISIT_BINDING_DISPATCH(DataflowVarNode);                                   \
    RELAX_VISIT_BINDING_DISPATCH(ShapeExprNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(ExternFuncNode);                                    \
    RELAX_VISIT_BINDING_DISPATCH(GlobalVarNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(FunctionNode);                                      \
    RELAX_VISIT_BINDING_DISPATCH(CallNode);                                          \
    RELAX_VISIT_BINDING_DISPATCH(SeqExprNode);                                       \
    RELAX_VISIT_BINDING_DISPATCH(IfNode);                                            \
    RELAX_VISIT_BINDING_DISPATCH(OpNode);                                            \
    RELAX_VISIT_BINDING_DISPATCH(TupleGetItemNode);                                  \
    return vtable;                                                                   \
  }                                                                                  \
  void Type::VisitBinding_(const VarBindingNode* binding) {                          \
    static VisitBindingVTable vtable = InitVisitBindingVTable();                     \
    const Expr& value = binding->value;                                              \
    ICHECK(value.defined()) << "Found null pointer node while traversing AST.";      \
    ICHECK(vtable.can_dispatch(value))                                               \
        << "VisitVarBinding do not allow binding value type" << value->GetTypeKey(); \
    vtable(value, this, binding);                                                    \
  }

// functions to be overriden.
#define RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(OP)                                   \
  void ExprVisitor::VisitBinding_(const VarBindingNode* binding, const OP* value) { \
    this->VisitExpr(binding->value);                                                \
    this->VisitVarDef(binding->var);                                                \
  }

// functions to be overriden.
#define RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(OP)                                   \
  void ExprMutator::VisitBinding_(const VarBindingNode* binding, const OP* value) { \
    Expr new_value = this->VisitExpr(binding->value);                               \
    this->ReEmitBinding(binding, new_value);                                        \
  }

namespace tvm {
namespace relax {

// ==================
// ExprVisitor

void ExprVisitor::VisitExpr(const Expr& expr) { ExprFunctor::VisitExpr(expr); }

void ExprVisitor::VisitExpr_(const ConstantNode* op) {
  this->VisitSpan(op->span);

  if (op->shape_) {
    this->VisitExpr(Downcast<Expr>(op->shape_.value()));
  }
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (Expr field : op->fields) {
    this->VisitExpr(field);
  }

  if (op->shape_) {
    this->VisitExpr(Downcast<Expr>(op->shape_.value()));
  }
}

// Visit the use-site of a defined Var
void ExprVisitor::VisitExpr_(const VarNode* op) { this->VisitSpan(op->span); }

// Visit the use-site of a defined DataflowVar
void ExprVisitor::VisitExpr_(const DataflowVarNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const FunctionNode* op) {
  this->VisitSpan(op->span);
  for (Var param : op->params) {
    this->VisitVarDef(param);
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

  if (op->shape_) {
    this->VisitExpr(Downcast<Expr>(op->shape_.value()));
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

void ExprVisitor::VisitExpr_(const ShapeExprNode* op) {
  for (PrimExpr val : op->values) {
    this->VisitPrimExpr(val);
  }
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const RuntimeDepShapeNode* op) { this->VisitSpan(op->span); }

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

void ExprVisitor::VisitPrimExpr(const PrimExpr& expr) {}

// implementations of binding visitor dispatch
RELAX_VAR_BINDING_DISPATCH_IMPL(ExprVisitor);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ConstantNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(TupleNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(VarNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(DataflowVarNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ShapeExprNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ExternFuncNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(GlobalVarNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(FunctionNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(CallNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(SeqExprNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(IfNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(OpNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(TupleGetItemNode);

void ExprVisitor::VisitBinding_(const MatchShapeNode* binding) {
  this->VisitExpr(binding->value);
  // TODO(ziheng): should we change pattern from
  // Array<PrimExpr> to ShapeExpr?
  this->VisitExpr(ShapeExpr(binding->pattern));
  if (binding->var.defined()) {
    this->VisitVarDef(binding->var);
  }
}

void ExprVisitor::VisitBindingBlock_(const BindingBlockNode* block) {
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitBindingBlock_(const DataflowBlockNode* block) {
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitVarDef_(const DataflowVarNode* var) {
  this->VisitSpan(var->span);

  if (var->shape_) {
    this->VisitExpr(Downcast<Expr>(var->shape_.value()));
  }
}

void ExprVisitor::VisitVarDef_(const VarNode* var) {
  this->VisitSpan(var->span);

  if (var->shape_) {
    this->VisitExpr(Downcast<Expr>(var->shape_.value()));
  }
}

void ExprVisitor::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchShapeNode>()) {
    VisitBinding_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
  }
}

void ExprVisitor::VisitBindingBlock(const BindingBlock& block) {
  if (const auto* node = block.as<DataflowBlockNode>()) {
    VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockNode>()) {
    VisitBindingBlock_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
  }
}

void ExprVisitor::VisitVarDef(const Var& var) {
  if (const auto* node = var.as<DataflowVarNode>()) {
    VisitVarDef_(node);
  } else if (const auto* node = var.as<VarNode>()) {
    VisitVarDef_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << var->GetTypeKey();
  }
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
// ExprMutatorBase

Expr ExprMutatorBase::VisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

Expr ExprMutatorBase::VisitExpr_(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const GlobalVarNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const TupleNode* op) {
  bool unchanged = true;
  tvm::Array<Expr> fields;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    unchanged &= new_field.same_as(field);
  }

  if (unchanged) {
    return GetRef<Expr>(op);
  } else {
    Expr new_tuple = Tuple(fields, op->span);
    return new_tuple;
  }
}

// Visit the use-site of a defined Var
Expr ExprMutatorBase::VisitExpr_(const VarNode* op) { return GetRef<Expr>(op); }

// Visit the use-site of a defined DataflowVar
Expr ExprMutatorBase::VisitExpr_(const DataflowVarNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const FunctionNode* op) {
  Expr body = this->VisitExpr(op->body);

  if (body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(op->params, body, op->ret_struct_info, op->attrs);
  }
}

Expr ExprMutatorBase::VisitExpr_(const CallNode* call_node) {
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
    Expr new_call = Call(new_op, call_args, call_node->attrs, ty_args, call_node->span);
    return new_call;
  }
}

Expr ExprMutatorBase::VisitExpr_(const IfNode* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitExpr(op->true_branch);
  Expr false_b = this->VisitExpr(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const TupleGetItemNode* op) {
  auto t = this->VisitExpr(op->tuple);
  if (op->tuple.same_as(t)) {
    return GetRef<Expr>(op);
  } else {
    return TupleGetItem(t, op->index, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const ShapeExprNode* op) {
  auto values = op->values.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

  if (values.same_as(op->values)) {
    return GetRef<Expr>(op);
  } else {
    return ShapeExpr(values, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const RuntimeDepShapeNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const ExternFuncNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  Expr body = this->VisitExpr(op->body);

  if (all_blocks_unchanged && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return SeqExpr(blocks, body);
  }
}

BindingBlock ExprMutatorBase::VisitBindingBlock(const BindingBlock& block) {
  Array<Binding> bindings;
  if (const auto* node = block.as<BindingBlockNode>()) {
    for (auto binding : node->bindings) {
      if (auto var_binding = binding.as<VarBindingNode>()) {
        Expr new_value = this->VisitExpr(var_binding->value);
        bindings.push_back(VarBinding(var_binding->var, new_value));
      } else if (auto match_shape_binding = binding.as<MatchShapeNode>()) {
        Expr new_value = this->VisitExpr(match_shape_binding->value);
        bindings.push_back(
            MatchShape(new_value, match_shape_binding->pattern, match_shape_binding->var));
      } else {
        LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
      }
    }
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
  }

  if (block.as<DataflowBlockNode>()) {
    return DataflowBlock(bindings);
  } else {
    return BindingBlock(bindings);
  }
}

Type ExprMutatorBase::VisitType(const Type& t) { return t; }

PrimExpr ExprMutatorBase::VisitPrimExpr(const PrimExpr& expr) { return expr; }

// ==================
// ExprMutator

Expr ExprMutator::VisitExpr(const Expr& expr) {
  return builder_->Normalize(ExprFunctor::VisitExpr(expr));
}

Expr ExprMutator::VisitExpr_(const TupleNode* op) {
  bool unchanged = true;
  tvm::Array<Expr> fields;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    unchanged &= new_field.same_as(field);
  }

  if (unchanged) {
    return GetRef<Expr>(op);
  } else {
    Expr new_tuple = Tuple(fields, op->span);
    return new_tuple;
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

  Expr body = this->VisitWithNewScope(op->body, params);

  if (all_params_unchanged && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(params, body, op->ret_struct_info, op->attrs);
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

RELAX_VAR_BINDING_DISPATCH_IMPL(ExprMutator);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ConstantNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(TupleNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(VarNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(DataflowVarNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ShapeExprNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ExternFuncNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(GlobalVarNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(FunctionNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(CallNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(SeqExprNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(IfNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(OpNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(TupleGetItemNode);

void ExprMutator::ReEmitBinding(const VarBindingNode* binding, Expr new_value) {
  Var new_var = this->VisitVarDef(binding->var);

  auto emit = [this](VarBinding b) {
    if (this->builder_->CurrentBlockIsDataFlow() && !b->var.as<DataflowVarNode>()) {
      this->builder_->EmitOutput(b);
    } else {
      this->builder_->Emit(b);
    }
  };

  // FIXME(@altanh): try to clean up all the fast paths and ty/shape infer, it's getting unwieldy
  // if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
  //   // no-op if there is no change
  //   emit(GetRef<VarBinding>(binding));
  //   return;
  // }

  // fast path: reemit binding if nothing changes
  if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
    emit(GetRef<VarBinding>(binding));
    return;
  }

  Var temp = WithStructInfo(new_var, GetStructInfo(new_value));
  if (!temp.same_as(new_var)) {
    new_var = temp;
    this->var_remap_[binding->var->vid] = new_var;
  }

  emit(VarBinding(new_var, new_value));
}

void ExprMutator::VisitBinding_(const MatchShapeNode* binding) {
  Expr new_value = this->VisitExpr(binding->value);
  Expr new_pattern = this->VisitExpr(ShapeExpr(binding->pattern));

  Var new_var;
  if (binding->var.defined()) {
    StructInfo new_sinfo = GetStructInfo(new_value);

    if (auto* ptr = new_sinfo.as<TensorStructInfoNode>()) {
      new_sinfo = TensorStructInfo(new_pattern, ptr->dtype);
    }
    new_var = this->VisitVarDef(binding->var);

    Var temp = WithStructInfo(new_var, new_sinfo);
    if (!temp.same_as(new_var)) {
      new_var = temp;
      this->var_remap_[binding->var->vid] = new_var;
    }
  }

  // reemit old binding if nothing changes
  if (new_value.same_as(binding->value) && new_pattern.same_as(binding->pattern)) {
    if (!binding->var.defined() || (binding->var.defined() && new_var.same_as(binding->var))) {
      builder_->EmitMatchShape(GetRef<MatchShape>(binding));
      return;
    }
  }

  // TODO(@altanh, @yuchen): shape and type inference here too...
  // TODO(@yuchen): when value's shape/type changed, create new var
  // TODO(@yuchen): group the can prove shape/type logic and replace var into a function
  builder_->EmitMatchShape(
      MatchShape(new_value, Downcast<ShapeExpr>(new_pattern)->values, new_var));
}

BindingBlock ExprMutator::VisitBindingBlock_(const BindingBlockNode* block) {
  builder_->BeginBindingBlock();
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

BindingBlock ExprMutator::VisitBindingBlock_(const DataflowBlockNode* block) {
  builder_->BeginDataflowBlock();
  for (auto binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

Var ExprMutator::VisitVarDef_(const DataflowVarNode* var) {
  // If an Expr have struct info, they must already be normalized,
  // This invariant is checked at the constructor location.
  // to simplify our overall development complexity and keep var def
  // stable by default.
  return GetRef<Var>(var);
}

Var ExprMutator::VisitVarDef_(const VarNode* var) { return GetRef<Var>(var); }

void ExprMutator::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchShapeNode>()) {
    VisitBinding_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
  }
}

BindingBlock ExprMutator::VisitBindingBlock(const BindingBlock& block) {
  BindingBlock ret;
  if (const auto* node = block.as<DataflowBlockNode>()) {
    ret = VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockNode>()) {
    ret = VisitBindingBlock_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
  }
  return ret;
}

Var ExprMutator::VisitVarDef(const Var& var) {
  Var ret;
  if (const auto* node = var.as<DataflowVarNode>()) {
    ret = VisitVarDef_(node);
  } else if (const auto* node = var.as<VarNode>()) {
    ret = VisitVarDef_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << var->GetTypeKey();
  }
  return ret;
}

Expr ExprMutator::VisitWithNewScope(const Expr& expr, Optional<Array<Var>> params) {
  ICHECK(expr->IsInstance<SeqExprNode>())
      << "Normal form requires all new scope is stored as SeqExpr";
  builder_->BeginScope(params);
  Expr ret = this->VisitExpr(expr);
  builder_->EndScope();
  return ret;
}

Optional<Expr> ExprMutator::LookupBinding(const Var& var) { return builder_->LookupBinding(var); }

Var ExprMutator::WithStructInfo(Var var, StructInfo struct_info) {
  ICHECK(struct_info.defined());

  // TODO(relax-team) add StructInfoEqual check
  if (var->struct_info_.defined()) {
    // use same-as as a quick path
    if (var->struct_info_.same_as(struct_info) ||
        StructuralEqual()(var->struct_info_, struct_info)) {
      return var;
    } else {
      Var new_var = var.as<DataflowVarNode>() ? DataflowVar(var->vid, struct_info, var->span)
                                              : Var(var->vid, struct_info, var->span);
      return new_var;
    }
  } else {
    UpdateStructInfo(var, struct_info);
    return var;
  }
}

TVM_REGISTER_GLOBAL("relax.MakePyExprVisitor").set_body_typed(PyExprVisitor::MakePyExprVisitor);

TVM_REGISTER_GLOBAL("relax.PyExprVisitorVisitExpr")
    .set_body_typed([](PyExprVisitor visitor, const Expr& expr) { visitor->VisitExpr(expr); });

TVM_REGISTER_GLOBAL("relax.PyExprVisitorVisitBinding")
    .set_body_typed([](PyExprVisitor visitor, const Binding& binding) {
      visitor->VisitBinding(binding);
    });

TVM_REGISTER_GLOBAL("relax.PyExprVisitorVisitBindingBlock")
    .set_body_typed([](PyExprVisitor visitor, const BindingBlock& block) {
      visitor->VisitBindingBlock(block);
    });

TVM_REGISTER_GLOBAL("relax.PyExprVisitorVisitVarDef")
    .set_body_typed([](PyExprVisitor visitor, const Var& var) { visitor->VisitVarDef(var); });

TVM_REGISTER_GLOBAL("relax.ExprVisitorVisitExpr")
    .set_body_typed([](PyExprVisitor visitor, const Expr& expr) {
      visitor->ExprVisitor::VisitExpr(expr);
    });

TVM_REGISTER_GLOBAL("relax.ExprVisitorVisitBinding")
    .set_body_typed([](PyExprVisitor visitor, const Binding& binding) {
      visitor->ExprVisitor::VisitBinding(binding);
    });

TVM_REGISTER_GLOBAL("relax.ExprVisitorVisitBindingBlock")
    .set_body_typed([](PyExprVisitor visitor, const BindingBlock& block) {
      visitor->ExprVisitor::VisitBindingBlock(block);
    });

TVM_REGISTER_GLOBAL("relax.ExprVisitorVisitVarDef")
    .set_body_typed([](PyExprVisitor visitor, const Var& var) {
      visitor->ExprVisitor::VisitVarDef(var);
    });

TVM_REGISTER_GLOBAL("relax.ExprVisitorVisitType")
    .set_body_typed([](PyExprVisitor visitor, const Type& type) {
      visitor->ExprVisitor::VisitType(type);
    });

TVM_REGISTER_GLOBAL("relax.ExprVisitorVisitSpan")
    .set_body_typed([](PyExprVisitor visitor, const Span& span) {
      visitor->ExprVisitor::VisitSpan(span);
    });

TVM_REGISTER_GLOBAL("relax.MakePyExprMutator").set_body_typed(PyExprMutator::MakePyExprMutator);

TVM_REGISTER_GLOBAL("relax.PyExprMutatorVisitExpr")
    .set_body_typed([](PyExprMutator mutator, const Expr& expr) {
      return mutator->VisitExpr(expr);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorVisitBinding")
    .set_body_typed([](PyExprMutator mutator, const Binding& binding) {
      mutator->VisitBinding(binding);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorVisitBindingBlock")
    .set_body_typed([](PyExprMutator mutator, const BindingBlock& block) {
      return mutator->VisitBindingBlock(block);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorVisitVarDef")
    .set_body_typed([](PyExprMutator mutator, const Var& var) {
      return mutator->VisitVarDef(var);
    });

TVM_REGISTER_GLOBAL("relax.ExprMutatorVisitExpr")
    .set_body_typed([](PyExprMutator mutator, const Expr& expr) {
      return mutator->ExprMutator::VisitExpr(expr);
    });

TVM_REGISTER_GLOBAL("relax.ExprMutatorVisitBinding")
    .set_body_typed([](PyExprMutator mutator, const Binding& binding) {
      return mutator->ExprMutator::VisitBinding(binding);
    });

TVM_REGISTER_GLOBAL("relax.ExprMutatorVisitBindingBlock")
    .set_body_typed([](PyExprMutator mutator, const BindingBlock& block) {
      return mutator->ExprMutator::VisitBindingBlock(block);
    });

TVM_REGISTER_GLOBAL("relax.ExprMutatorVisitVarDef")
    .set_body_typed([](PyExprMutator mutator, const Var& var) {
      return mutator->ExprMutator::VisitVarDef(var);
    });

TVM_REGISTER_GLOBAL("relax.ExprMutatorVisitType")
    .set_body_typed([](PyExprMutator mutator, const Type& type) {
      return mutator->ExprMutator::VisitType(type);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorVisitExprPostOrder")
    .set_body_typed([](PyExprMutator mutator, const Expr& expr) {
      return mutator->VisitExprPostOrder(expr);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorVisitWithNewScope")
    .set_body_typed([](PyExprMutator mutator, const Expr& expr) {
      return mutator->VisitWithNewScope(expr);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorLookupBinding")
    .set_body_typed([](PyExprMutator mutator, const Var& var) {
      return mutator->LookupBinding(var);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorWithStructInfo")
    .set_body_typed([](PyExprMutator mutator, Var var, StructInfo sinfo) {
      return mutator->WithStructInfo(var, sinfo);
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorSetVarRemap")
    .set_body_typed([](PyExprMutator mutator, Id id, Var var) {
      return mutator->var_remap_[id] = var;
    });

TVM_REGISTER_GLOBAL("relax.PyExprMutatorGetVarRemap")
    .set_body_typed([](PyExprMutator mutator, Id id) { return mutator->var_remap_[id]; });

}  // namespace relax
}  // namespace tvm
