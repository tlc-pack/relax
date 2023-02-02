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
    RELAX_VISIT_BINDING_DISPATCH(PrimValueNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(StringImmNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(DataTypeImmNode);                                   \
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

void ExprVisitor::VisitExprDepStructInfoField(const StructInfo& struct_info) {
  // recurse into struct info in case they depend on value
  // under the current scope.
  default_struct_info_field_visitor_.VisitStructInfo(struct_info);
}

ExprVisitor::DefaultStructInfoFieldVisitor::DefaultStructInfoFieldVisitor(ExprVisitor* parent)
    : parent_(parent) {}

void ExprVisitor::DefaultStructInfoFieldVisitor::VisitStructInfoExprField(const Expr& expr) {
  parent_->VisitExpr(expr);
}

void ExprVisitor::DefaultStructInfoFieldVisitor::VisitStructInfoExprField(const PrimExpr& expr) {
  parent_->VisitPrimExpr(expr);
}

void ExprVisitor::DefaultStructInfoFieldVisitor::VisitStructInfo_(const FuncStructInfoNode* op) {
  // Do not recurse into function struct info
  // as they won't contain ref to values in current scope.
}

void ExprVisitor::VisitExpr(const Expr& expr) { ExprFunctor::VisitExpr(expr); }

void ExprVisitor::VisitExpr_(const ConstantNode* op) {
  this->VisitSpan(op->span);
  // Constant's StructInfo does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) {
  this->VisitSpan(op->span);
  // FuncStructInfo is not value-dep
}

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (Expr field : op->fields) {
    this->VisitExpr(field);
  }
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

// Visit the use-site of a defined Var
void ExprVisitor::VisitExpr_(const VarNode* op) {
  this->VisitSpan(op->span);
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

// Visit the use-site of a defined DataflowVar
void ExprVisitor::VisitExpr_(const DataflowVarNode* op) {
  this->VisitSpan(op->span);
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const FunctionNode* op) {
  this->VisitSpan(op->span);
  for (Var param : op->params) {
    this->VisitVarDef(param);
  }

  this->VisitExpr(op->body);
  // FuncStructInfo does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (StructInfo sinfo_arg : op->sinfo_args) {
    this->VisitExprDepStructInfoField(sinfo_arg);
  }

  for (Expr arg : op->args) {
    this->VisitExpr(arg);
  }

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const OpNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->tuple);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const ShapeExprNode* op) {
  for (PrimExpr val : op->values) {
    this->VisitPrimExpr(val);
  }
  this->VisitSpan(op->span);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const ExternFuncNode* op) {
  this->VisitSpan(op->span);
  // FuncStructInfo does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const SeqExprNode* op) {
  this->VisitSpan(op->span);
  for (BindingBlock block : op->blocks) {
    this->VisitBindingBlock(block);
  }
  this->VisitExpr(op->body);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const PrimValueNode* op) {
  this->VisitPrimExpr(op->value);
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const StringImmNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const DataTypeImmNode* op) { this->VisitSpan(op->span); }

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
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(PrimValueNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(StringImmNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(DataTypeImmNode);

void ExprVisitor::VisitBinding_(const MatchCastNode* binding) {
  this->VisitExpr(binding->value);
  this->VisitVarDef(binding->var);
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

void ExprVisitor::VisitVarDef_(const DataflowVarNode* var) { this->VisitSpan(var->span); }

void ExprVisitor::VisitVarDef_(const VarNode* var) { this->VisitSpan(var->span); }

void ExprVisitor::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastNode>()) {
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

StructInfo ExprMutatorBase::VisitExprDepStructInfoField(const StructInfo& struct_info) {
  // recurse into struct info in case they depend on value
  // under the current scope.
  return default_struct_info_field_mutator_.VisitStructInfo(struct_info);
}

ExprMutatorBase::DefaultStructInfoFieldMutator::DefaultStructInfoFieldMutator(
    ExprMutatorBase* parent)
    : parent_(parent) {}

Expr ExprMutatorBase::DefaultStructInfoFieldMutator::VisitStructInfoExprField(const Expr& expr) {
  return parent_->VisitExpr(expr);
}

PrimExpr ExprMutatorBase::DefaultStructInfoFieldMutator::VisitStructInfoExprField(
    const PrimExpr& expr) {
  return parent_->VisitPrimExpr(expr);
}

StructInfo ExprMutatorBase::DefaultStructInfoFieldMutator::VisitStructInfo_(
    const FuncStructInfoNode* op) {
  // Do not recurse into function struct info
  // as they won't contain ref to values in current scope.
  return GetRef<StructInfo>(op);
}

Expr ExprMutatorBase::VisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

Expr ExprMutatorBase::VisitExpr_(const ConstantNode* op) {
  // Constant' struct info won't be affected by Expr/PrimExpr change.
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const GlobalVarNode* op) {
  // FuncStructInfo won't be affected by Expr/PrimExpr change.
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const TupleNode* op) {
  bool unchanged = true;
  tvm::Array<Expr> fields;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    unchanged &= new_field.same_as(field);
  }

  if (unchanged) {
    // If tuple's struct info change it means that
    // one of its fields' struct info will change
    // so un-changed already implies that struct info won't change
    return GetRef<Expr>(op);
  } else {
    // when there is a change return a new tuple node
    return Tuple(fields, op->span);
  }
}

// Visit the use-site of a defined Var
Expr ExprMutatorBase::VisitExpr_(const VarNode* op) {
  // struct info of var-use should remain stable
  // or the var itself will get replaced
  return GetRef<Expr>(op);
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutatorBase::VisitExpr_(const DataflowVarNode* op) {
  // struct info of var-use should remain stable
  // or the var itself will get replaced
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const FunctionNode* op) {
  // struct info of function is not value dependent
  // so no need to check struct_info field
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

  Array<StructInfo> sinfo_args;
  for (StructInfo sinfo_arg : call_node->sinfo_args) {
    StructInfo new_sinfo_arg = this->VisitExprDepStructInfoField(sinfo_arg);
    sinfo_args.push_back(new_sinfo_arg);
    unchanged &= new_sinfo_arg.same_as(sinfo_arg);
  }

  tvm::Array<Expr> call_args;
  for (Expr arg : call_node->args) {
    Expr new_arg = this->VisitExpr(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  if (unchanged && VisitAndCheckStructInfoFieldUnchanged(call_node->struct_info_)) {
    return GetRef<Expr>(call_node);
  } else {
    return Call(new_op, call_args, call_node->attrs, sinfo_args, call_node->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const IfNode* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitExpr(op->true_branch);
  Expr false_b = this->VisitExpr(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const TupleGetItemNode* op) {
  auto t = this->VisitExpr(op->tuple);
  if (op->tuple.same_as(t)) {
    // struct info can be deterministically derived by tuple and index
    // if t does not change, then struct info won't change.
    return GetRef<Expr>(op);
  } else {
    return TupleGetItem(t, op->index, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const PrimValueNode* op) {
  auto value = this->VisitPrimExpr(op->value);
  if (op->value.same_as(value)) {
    // struct info can be deterministically derived by value
    // if value does not change, then struct info won't change.
    return GetRef<Expr>(op);
  }
  return PrimValue(value, op->span);
}

Expr ExprMutatorBase::VisitExpr_(const StringImmNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const DataTypeImmNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const ShapeExprNode* op) {
  auto values = op->values.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

  if (values.same_as(op->values)) {
    // If values does not change, struct info won't change.
    return GetRef<Expr>(op);
  } else {
    return ShapeExpr(values, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const ExternFuncNode* op) {
  // StructInfo of function remains value independent.
  return GetRef<Expr>(op);
}

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

  if (all_blocks_unchanged && body.same_as(op->body) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
    return GetRef<Expr>(op);
  }
  return SeqExpr(blocks, body);
}

BindingBlock ExprMutatorBase::VisitBindingBlock(const BindingBlock& block) {
  Array<Binding> bindings;
  if (const auto* node = block.as<BindingBlockNode>()) {
    for (auto binding : node->bindings) {
      if (auto var_binding = binding.as<VarBindingNode>()) {
        Expr new_value = this->VisitExpr(var_binding->value);
        bindings.push_back(VarBinding(var_binding->var, new_value));
      } else if (auto match_cast = binding.as<MatchCastNode>()) {
        Expr new_value = this->VisitExpr(match_cast->value);
        bindings.push_back(MatchCast(match_cast->var, new_value, match_cast->struct_info));
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

PrimExpr ExprMutatorBase::VisitPrimExpr(const PrimExpr& expr) { return expr; }

// ==================
// ExprMutator

Expr ExprMutator::VisitExpr(const Expr& expr) {
  return builder_->Normalize(ExprFunctor::VisitExpr(expr));
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

  // FuncStructInfo does not depend on Expr
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
      op->false_branch.same_as(false_b) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
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

  if (all_blocks_unchanged && body.same_as(op->body) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
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
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(PrimValueNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(StringImmNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(DataTypeImmNode);

void ExprMutator::ReEmitBinding(const VarBindingNode* binding, Expr new_value) {
  Var new_var = this->VisitVarDef(binding->var);

  // fast path: reemit binding if nothing changes
  if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
    builder_->EmitNormalized(GetRef<VarBinding>(binding));
    return;
  }

  Var temp = WithStructInfo(new_var, GetStructInfo(new_value));
  if (!temp.same_as(new_var)) {
    new_var = temp;
    this->var_remap_[binding->var->vid] = new_var;
  }

  builder_->EmitNormalized(VarBinding(new_var, new_value));
}

void ExprMutator::VisitBinding_(const MatchCastNode* binding) {
  Var new_var = this->VisitVarDef(binding->var);
  Expr new_value = this->VisitExpr(binding->value);

  // re-emit old binding if nothing changes
  if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
    builder_->EmitNormalized(GetRef<MatchCast>(binding));
  } else {
    new_value = builder_->NormalizeArgument(new_value);
    builder_->EmitNormalized(MatchCast(new_var, new_value, binding->struct_info, binding->span));
  }
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
  if (auto* sinfo = var->struct_info_.as<StructInfoNode>()) {
    StructInfo struct_info = this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    if (struct_info.same_as(var->struct_info_)) {
      return GetRef<DataflowVar>(var);
    } else {
      return DataflowVar(var->vid, struct_info, var->span);
    }
  } else {
    return GetRef<DataflowVar>(var);
  }
}

Var ExprMutator::VisitVarDef_(const VarNode* var) {
  if (auto* sinfo = var->struct_info_.as<StructInfoNode>()) {
    StructInfo struct_info = this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    if (struct_info.same_as(var->struct_info_)) {
      return GetRef<Var>(var);
    } else {
      return Var(var->vid, struct_info, var->span);
    }
  } else {
    return GetRef<Var>(var);
  }
}

void ExprMutator::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastNode>()) {
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

/*!
 * \brief The abstract interface of ExprVisitor.
 */
class PyExprVisitorNode : public Object, public ExprVisitor {
 private:
  using TSelf = PyExprVisitorNode;
  using FType = tvm::NodeFunctor<void(const ObjectRef& n, TSelf* self)>;

 public:
  /*! \brief The packed function to the `VisitExpr(const Expr& expr)` function. */
  PackedFunc f_visit_expr{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ConstantNode* op)` function. */
  PackedFunc f_visit_constant_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleNode* op)` function. */
  PackedFunc f_visit_tuple_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const VarNode* op)` function. */
  PackedFunc f_visit_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataflowVarNode* op)` function. */
  PackedFunc f_visit_dataflow_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ShapeExprNode* op)` function. */
  PackedFunc f_visit_shape_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ExternFuncNode* op)` function. */
  PackedFunc f_visit_extern_func_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GlobalVarNode* op)` function. */
  PackedFunc f_visit_global_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FunctionNode* op)` function. */
  PackedFunc f_visit_function_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CallNode* op)` function. */
  PackedFunc f_visit_call_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SeqExprNode* op)` function. */
  PackedFunc f_visit_seq_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const IfNode* op)` function. */
  PackedFunc f_visit_if_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const OpNode* op)` function. */
  PackedFunc f_visit_op_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleGetItemNode* op)` function. */
  PackedFunc f_visit_tuple_getitem_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const PrimValueNode* op)` function. */
  PackedFunc f_visit_prim_value_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const StringImmNode* op)` function. */
  PackedFunc f_visit_string_imm_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataTypeImmNode* op)` function. */
  PackedFunc f_visit_data_type_imm_{nullptr};
  /*! \brief The packed function to the `VisitBinding(const Binding& binding)` function. */
  PackedFunc f_visit_binding{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const VarBindingNode* binding)`
   * function. */
  PackedFunc f_visit_var_binding_{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const MatchCastNode* binding)`
   * function. */
  PackedFunc f_visit_match_cast_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock(const BindingBlock& block)`
   * function. */
  PackedFunc f_visit_binding_block{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const BindingBlockNode* block)`
   * function. */
  PackedFunc f_visit_binding_block_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const DataflowBlockNode* block)`
   * function. */
  PackedFunc f_visit_dataflow_block_{nullptr};
  /*! \brief The packed function to the `VisitVarDef(const Var& var)` function. */
  PackedFunc f_visit_var_def{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const VarNode* var)` function. */
  PackedFunc f_visit_var_def_{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const DataflowVarNode* var)` function. */
  PackedFunc f_visit_dataflow_var_def_{nullptr};
  /*! \brief The packed function to the `VisitSpan(const Span& span)` function. */
  PackedFunc f_visit_span{nullptr};

  void VisitExpr(const Expr& expr) {
    if (f_visit_expr != nullptr) {
      f_visit_expr(expr);
    } else {
      // Need to init the overwrite VTable
      static FType vtable = InitVTable();
      vtable(expr, this);
    }
  }

  void VisitBinding(const Binding& binding)
      PY_EXPR_VISITOR_DEFAULT(binding, f_visit_binding, ExprVisitor::VisitBinding(binding));

  void VisitBinding_(const VarBindingNode* binding)
      PY_EXPR_VISITOR_DEFAULT(GetRef<VarBinding>(binding), f_visit_var_binding_,
                              ExprVisitor::VisitBinding_(binding));
  void VisitBinding_(const MatchCastNode* binding)
      PY_EXPR_VISITOR_DEFAULT(GetRef<MatchCast>(binding), f_visit_match_cast_,
                              ExprVisitor::VisitBinding_(binding));

  void VisitBindingBlock(const BindingBlock& block)
      PY_EXPR_VISITOR_DEFAULT(block, f_visit_binding_block, ExprVisitor::VisitBindingBlock(block));

  void VisitBindingBlock_(const BindingBlockNode* block)
      PY_EXPR_VISITOR_DEFAULT(GetRef<BindingBlock>(block), f_visit_binding_block_,
                              ExprVisitor::VisitBindingBlock_(block));
  void VisitBindingBlock_(const DataflowBlockNode* block)
      PY_EXPR_VISITOR_DEFAULT(GetRef<DataflowBlock>(block), f_visit_dataflow_block_,
                              ExprVisitor::VisitBindingBlock_(block));

  void VisitVarDef(const Var& var)
      PY_EXPR_VISITOR_DEFAULT(var, f_visit_var_def, ExprVisitor::VisitVarDef(var));
  void VisitVarDef_(const VarNode* var)
      PY_EXPR_VISITOR_DEFAULT(GetRef<Var>(var), f_visit_var_def_, ExprVisitor::VisitVarDef_(var));
  void VisitVarDef_(const DataflowVarNode* var)
      PY_EXPR_VISITOR_DEFAULT(GetRef<DataflowVar>(var), f_visit_dataflow_var_def_,
                              ExprVisitor::VisitVarDef_(var));

  void VisitSpan(const Span& span)
      PY_EXPR_VISITOR_DEFAULT(span, f_visit_span, ExprVisitor::VisitSpan(span));

  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "expr_functor.PyExprVisitor";
  TVM_DECLARE_BASE_OBJECT_INFO(PyExprVisitorNode, Object);

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    PY_EXPR_VISITOR_DISPATCH(ConstantNode, f_visit_constant_);
    PY_EXPR_VISITOR_DISPATCH(TupleNode, f_visit_tuple_);
    PY_EXPR_VISITOR_DISPATCH(VarNode, f_visit_var_);
    PY_EXPR_VISITOR_DISPATCH(DataflowVarNode, f_visit_dataflow_var_);
    PY_EXPR_VISITOR_DISPATCH(ShapeExprNode, f_visit_shape_expr_);
    PY_EXPR_VISITOR_DISPATCH(ExternFuncNode, f_visit_extern_func_);
    PY_EXPR_VISITOR_DISPATCH(GlobalVarNode, f_visit_global_var_);
    PY_EXPR_VISITOR_DISPATCH(FunctionNode, f_visit_function_);
    PY_EXPR_VISITOR_DISPATCH(CallNode, f_visit_call_);
    PY_EXPR_VISITOR_DISPATCH(SeqExprNode, f_visit_seq_expr_);
    PY_EXPR_VISITOR_DISPATCH(IfNode, f_visit_if_);
    PY_EXPR_VISITOR_DISPATCH(OpNode, f_visit_op_);
    PY_EXPR_VISITOR_DISPATCH(TupleGetItemNode, f_visit_tuple_getitem_);
    PY_EXPR_VISITOR_DISPATCH(PrimValueNode, f_visit_prim_value_);
    PY_EXPR_VISITOR_DISPATCH(StringImmNode, f_visit_string_imm_);
    PY_EXPR_VISITOR_DISPATCH(DataTypeImmNode, f_visit_data_type_imm_);
    return vtable;
  }
};

TVM_REGISTER_NODE_TYPE(PyExprVisitorNode);

/*!
 * \brief Managed reference to PyExprVisitorNode.
 * \sa PyExprVisitorNode
 */
class PyExprVisitor : public ObjectRef {
 public:
  /*!
   * \brief Create a PyExprVisitor with customized methods on the python-side.
   * \param f_visit_expr The packed function of `VisitExpr(const Expr& expr)`.
   * \param f_visit_constant_ The packed function of `VisitExpr_(const ConstantNode* op)`.
   * \param f_visit_tuple_ The packed function of `VisitExpr_(const TupleNode* op)`.
   * \param f_visit_var_ The packed function of `VisitExpr_(const VarNode* op)`.
   * \param f_visit_dataflow_var_ The packed function of `VisitExpr_(const DataflowVarNode* op)`.
   * \param f_visit_shape_expr_ The packed function of `VisitExpr_(const ShapeExprNode* op)`.
   * \param f_visit_extern_func_ The packed function of `VisitExpr_(const ExternFuncNode* op)`.
   * \param f_visit_global_var_ The packed function of `VisitExpr_(const GlobalVarNode* op)`.
   * \param f_visit_function_ The packed function of `VisitExpr_(const FunctionNode* op)`.
   * \param f_visit_call_ The packed function of `VisitExpr_(const CallNode* op)`.
   * \param f_visit_seq_expr_ The packed function of `VisitExpr_(const SeqExprNode* op)`.
   * \param f_visit_if_ The packed function of `VisitExpr_(const IfNode* op)`.
   * \param f_visit_op_ The packed function of `VisitExpr_(const OpNode* op)`.
   * \param f_visit_tuple_getitem_ The packed function of `VisitExpr_(const TupleGetItemNode* op)`.
   * \param f_visit_prim_value_ The packed function of `VisitExpr_(const PrimValueNode* op)`.
   * \param f_visit_string_imm_ The packed function of `VisitExpr_(const StringImmNode* op)`.
   * \param f_visit_data_type_imm_ The packed function of `VisitExpr_(const DataTypeImmNode* op)`.
   * \param f_visit_binding The packed function of `VisitBinding(const Binding& binding)`.
   * \param f_visit_var_binding_ The packed function of `VisitBinding_(const VarBindingNode*
   * binding)`.
   * \param f_visit_match_cast_ The packed function of `VisitBinding_(const MatchCastNode*
   * binding)`.
   * \param f_visit_binding_block The packed function of `VisitBindingBlock(const BindingBlock&
   * block)`.
   * \param f_visit_binding_block_ The packed function of `VisitBindingBlock_(const
   * BindingBlockNode* block)`.
   * \param f_visit_dataflow_block_ The packed function of `VisitBindingBlock_(const
   * DataflowBlockNode* block)`.
   * \param f_visit_var_def The packed function of `VisitVarDef(const Var& var)`.
   * \param f_visit_var_def_ The packed function of `VisitVarDef_(const VarNode* var)`.
   * \param f_visit_dataflow_var_def_ The packed function of `VisitVarDef_(const DataflowVarNode*
   * var)`.
   * \param f_visit_span The packed function of `VisitSpan(const Span& span)`.
   * \return The PyVisitor created.
   */
  TVM_DLL static PyExprVisitor MakePyExprVisitor(
      PackedFunc f_visit_expr, PackedFunc f_visit_constant_, PackedFunc f_visit_tuple_,
      PackedFunc f_visit_var_, PackedFunc f_visit_dataflow_var_, PackedFunc f_visit_shape_expr_,
      PackedFunc f_visit_extern_func_, PackedFunc f_visit_global_var_, PackedFunc f_visit_function_,
      PackedFunc f_visit_call_, PackedFunc f_visit_seq_expr_, PackedFunc f_visit_if_,
      PackedFunc f_visit_op_, PackedFunc f_visit_tuple_getitem_, PackedFunc f_visit_prim_value_,
      PackedFunc f_visit_string_imm_, PackedFunc f_visit_data_type_imm_, PackedFunc f_visit_binding,
      PackedFunc f_visit_var_binding_, PackedFunc f_visit_match_cast_,
      PackedFunc f_visit_binding_block, PackedFunc f_visit_binding_block_,
      PackedFunc f_visit_dataflow_block_, PackedFunc f_visit_var_def, PackedFunc f_visit_var_def_,
      PackedFunc f_visit_dataflow_var_def_, PackedFunc f_visit_span) {
    ObjectPtr<PyExprVisitorNode> n = make_object<PyExprVisitorNode>();
    n->f_visit_expr = f_visit_expr;
    n->f_visit_binding = f_visit_binding;
    n->f_visit_binding_block = f_visit_binding_block;
    n->f_visit_var_def = f_visit_var_def;
    n->f_visit_span = f_visit_span;
    n->f_visit_constant_ = f_visit_constant_;
    n->f_visit_tuple_ = f_visit_tuple_;
    n->f_visit_var_ = f_visit_var_;
    n->f_visit_dataflow_var_ = f_visit_dataflow_var_;
    n->f_visit_shape_expr_ = f_visit_shape_expr_;
    n->f_visit_extern_func_ = f_visit_extern_func_;
    n->f_visit_global_var_ = f_visit_global_var_;
    n->f_visit_function_ = f_visit_function_;
    n->f_visit_call_ = f_visit_call_;
    n->f_visit_seq_expr_ = f_visit_seq_expr_;
    n->f_visit_if_ = f_visit_if_;
    n->f_visit_op_ = f_visit_op_;
    n->f_visit_tuple_getitem_ = f_visit_tuple_getitem_;
    n->f_visit_prim_value_ = f_visit_prim_value_;
    n->f_visit_string_imm_ = f_visit_string_imm_;
    n->f_visit_data_type_imm_ = f_visit_data_type_imm_;
    n->f_visit_var_binding_ = f_visit_var_binding_;
    n->f_visit_match_cast_ = f_visit_match_cast_;
    n->f_visit_binding_block_ = f_visit_binding_block_;
    n->f_visit_dataflow_block_ = f_visit_dataflow_block_;
    n->f_visit_var_def_ = f_visit_var_def_;
    n->f_visit_dataflow_var_def_ = f_visit_dataflow_var_def_;
    return PyExprVisitor(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyExprVisitor, ObjectRef, PyExprVisitorNode);
};

/*!
 * \brief The abstract interface of ExprMutator.
 */
class PyExprMutatorNode : public Object, public ExprMutator {
 private:
  using TSelf = PyExprMutatorNode;
  using FType = tvm::NodeFunctor<Expr(const ObjectRef& n, TSelf* self)>;

 public:
  /*! \brief The packed function to the `VisitExpr(const Expr& expr)` function. */
  PackedFunc f_visit_expr{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ConstantNode* op)` function. */
  PackedFunc f_visit_constant_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleNode* op)` function. */
  PackedFunc f_visit_tuple_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const VarNode* op)` function. */
  PackedFunc f_visit_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataflowVarNode* op)` function. */
  PackedFunc f_visit_dataflow_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ShapeExprNode* op)` function. */
  PackedFunc f_visit_shape_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ExternFuncNode* op)` function. */
  PackedFunc f_visit_extern_func_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GlobalVarNode* op)` function. */
  PackedFunc f_visit_global_var_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FunctionNode* op)` function. */
  PackedFunc f_visit_function_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CallNode* op)` function. */
  PackedFunc f_visit_call_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SeqExprNode* op)` function. */
  PackedFunc f_visit_seq_expr_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const IfNode* op)` function. */
  PackedFunc f_visit_if_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const OpNode* op)` function. */
  PackedFunc f_visit_op_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const TupleGetItemNode* op)` function. */
  PackedFunc f_visit_tuple_getitem_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const PrimValueNode* op)` function. */
  PackedFunc f_visit_prim_value_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const StringImmNode* op)` function. */
  PackedFunc f_visit_string_imm_{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DataTypeImmNode* op)` function. */
  PackedFunc f_visit_data_type_imm_{nullptr};
  /*! \brief The packed function to the `VisitBinding(const Binding& binding)` function. */
  PackedFunc f_visit_binding{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const VarBindingNode* binding)`
   * function. */
  PackedFunc f_visit_var_binding_{nullptr};
  /*! \brief The packed function to the `VisitBinding_(const MatchCastNode* binding)`
   * function. */
  PackedFunc f_visit_match_cast_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock(const BindingBlock& block)`
   * function. */
  PackedFunc f_visit_binding_block{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const BindingBlockNode* block)`
   * function. */
  PackedFunc f_visit_binding_block_{nullptr};
  /*! \brief The packed function to the `VisitBindingBlock_(const DataflowBlockNode* block)`
   * function. */
  PackedFunc f_visit_dataflow_block_{nullptr};
  /*! \brief The packed function to the `VisitVarDef(const Var& var)` function. */
  PackedFunc f_visit_var_def{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const VarNode* var)` function. */
  PackedFunc f_visit_var_def_{nullptr};
  /*! \brief The packed function to the `VisitVarDef_(const DataflowVarNode* var)` function. */
  PackedFunc f_visit_dataflow_var_def_{nullptr};
  /*! \brief The packed function to the `VisitSpan(const Span& span)` function. */
  PackedFunc f_visit_span{nullptr};

  Expr VisitExpr(const Expr& expr) {
    if (f_visit_expr != nullptr) {
      return builder_->Normalize(f_visit_expr(expr));
    } else {
      static FType vtable = InitVTable();
      return builder_->Normalize(vtable(expr, this));
    }
  }

  void VisitBinding(const Binding& binding) {
    if (f_visit_binding != nullptr)
      f_visit_binding(binding);
    else
      ExprMutator::VisitBinding(binding);
  }

  void VisitBinding_(const VarBindingNode* binding) {
    if (f_visit_var_binding_ != nullptr)
      f_visit_var_binding_(GetRef<VarBinding>(binding));
    else
      ExprMutator::VisitBinding_(binding);
  }

  void VisitBinding_(const MatchCastNode* binding) {
    if (f_visit_match_cast_ != nullptr)
      f_visit_match_cast_(GetRef<MatchCast>(binding));
    else
      ExprMutator::VisitBinding_(binding);
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block)
      PY_EXPR_MUTATOR_DEFAULT(block, f_visit_binding_block, ExprMutator::VisitBindingBlock(block),
                              BindingBlock);

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block)
      PY_EXPR_MUTATOR_DEFAULT(GetRef<BindingBlock>(block), f_visit_binding_block_,
                              ExprMutator::VisitBindingBlock_(block), BindingBlock);
  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block)
      PY_EXPR_MUTATOR_DEFAULT(GetRef<DataflowBlock>(block), f_visit_dataflow_block_,
                              ExprMutator::VisitBindingBlock_(block), BindingBlock);

  Var VisitVarDef(const Var& var)
      PY_EXPR_MUTATOR_DEFAULT(var, f_visit_var_def, ExprMutator::VisitVarDef(var), Var);
  Var VisitVarDef_(const VarNode* var) PY_EXPR_MUTATOR_DEFAULT(GetRef<Var>(var), f_visit_var_def_,
                                                               ExprMutator::VisitVarDef_(var), Var);
  Var VisitVarDef_(const DataflowVarNode* var)
      PY_EXPR_MUTATOR_DEFAULT(GetRef<DataflowVar>(var), f_visit_dataflow_var_def_,
                              ExprMutator::VisitVarDef_(var), Var);

  /*!
   * \brief Dispatcher for post-order rewrite.
   * \param expr The Expr to be rewritten.
   * \return The Expr after post-order rewritten.
   */
  Expr VisitExprPostOrder(const Expr& expr) {
    static FType post_order_vtable = InitPostOrderVTable();
    return post_order_vtable(expr, this);
  }

  using ExprMutator::builder_;
  using ExprMutator::LookupBinding;
  using ExprMutator::var_remap_;
  using ExprMutator::VisitWithNewScope;
  using ExprMutator::WithStructInfo;

  void VisitAttrs(AttrVisitor* v) { v->Visit("builder_", &builder_); }
  static constexpr const char* _type_key = "expr_functor.PyExprMutator";
  TVM_DECLARE_BASE_OBJECT_INFO(PyExprMutatorNode, Object);

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    PY_EXPR_MUTATOR_DISPATCH(ConstantNode, f_visit_constant_);
    PY_EXPR_MUTATOR_DISPATCH(TupleNode, f_visit_tuple_);
    PY_EXPR_MUTATOR_DISPATCH(VarNode, f_visit_var_);
    PY_EXPR_MUTATOR_DISPATCH(DataflowVarNode, f_visit_dataflow_var_);
    PY_EXPR_MUTATOR_DISPATCH(ShapeExprNode, f_visit_shape_expr_);
    PY_EXPR_MUTATOR_DISPATCH(ExternFuncNode, f_visit_extern_func_);
    PY_EXPR_MUTATOR_DISPATCH(GlobalVarNode, f_visit_global_var_);
    PY_EXPR_MUTATOR_DISPATCH(FunctionNode, f_visit_function_);
    PY_EXPR_MUTATOR_DISPATCH(CallNode, f_visit_call_);
    PY_EXPR_MUTATOR_DISPATCH(SeqExprNode, f_visit_seq_expr_);
    PY_EXPR_MUTATOR_DISPATCH(IfNode, f_visit_if_);
    PY_EXPR_MUTATOR_DISPATCH(OpNode, f_visit_op_);
    PY_EXPR_MUTATOR_DISPATCH(TupleGetItemNode, f_visit_tuple_getitem_);
    PY_EXPR_MUTATOR_DISPATCH(PrimValueNode, f_visit_prim_value_);
    PY_EXPR_MUTATOR_DISPATCH(StringImmNode, f_visit_string_imm_);
    PY_EXPR_MUTATOR_DISPATCH(DataTypeImmNode, f_visit_data_type_imm_);
    return vtable;
  }

  // initialize the vtable for post order visit.
  static FType InitPostOrderVTable() {
    FType post_order_vtable;
    // Set dispatch
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(ConstantNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(TupleNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(VarNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(DataflowVarNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(ShapeExprNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(ExternFuncNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(GlobalVarNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(FunctionNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(CallNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(SeqExprNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(IfNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(OpNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(TupleGetItemNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(PrimValueNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(StringImmNode);
    PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(DataTypeImmNode);
    return post_order_vtable;
  }
};

TVM_REGISTER_NODE_TYPE(PyExprMutatorNode);

/*!
 * \brief Managed reference to PyExprMutatorNode.
 * \sa PyExprMutatorNode
 */
class PyExprMutator : public ObjectRef {
 public:
  /*!
   * \brief Create a PyExprMutator with customized methods on the python-side.
   * \param f_visit_expr The packed function of `VisitExpr(const Expr& expr)`.
   * \param f_visit_constant_ The packed function of `VisitExpr_(const ConstantNode* op)`.
   * \param f_visit_tuple_ The packed function of `VisitExpr_(const TupleNode* op)`.
   * \param f_visit_var_ The packed function of `VisitExpr_(const VarNode* op)`.
   * \param f_visit_dataflow_var_ The packed function of `VisitExpr_(const DataflowVarNode* op)`.
   * \param f_visit_shape_expr_ The packed function of `VisitExpr_(const ShapeExprNode* op)`.
   * \param f_visit_extern_func_ The packed function of `VisitExpr_(const ExternFuncNode* op)`.
   * \param f_visit_global_var_ The packed function of `VisitExpr_(const GlobalVarNode* op)`.
   * \param f_visit_function_ The packed function of `VisitExpr_(const FunctionNode* op)`.
   * \param f_visit_call_ The packed function of `VisitExpr_(const CallNode* op)`.
   * \param f_visit_seq_expr_ The packed function of `VisitExpr_(const SeqExprNode* op)`.
   * \param f_visit_if_ The packed function of `VisitExpr_(const IfNode* op)`.
   * \param f_visit_op_ The packed function of `VisitExpr_(const OpNode* op)`.
   * \param f_visit_tuple_getitem_ The packed function of `VisitExpr_(const TupleGetItemNode* op)`.
   * \param f_visit_prim_value_ The packed function of `VisitExpr_(const PrimValueNode* op)`.
   * \param f_visit_string_imm_ The packed function of `VisitExpr_(const StringImmNode* op)`.
   * \param f_visit_data_type_imm_ The packed function of `VisitExpr_(const DataTypeImmNode* op)`.
   * \param f_visit_binding The packed function of `VisitBinding(const Binding& binding)`.
   * \param f_visit_var_binding_ The packed function of `VisitBinding_(const VarBindingNode*
   * binding)`.
   * \param f_visit_match_cast_ The packed function of `VisitBinding_(const MatchCastNode*
   * binding)`.
   * \param f_visit_binding_block The packed function of `VisitBindingBlock(const BindingBlock&
   * block)`.
   * \param f_visit_binding_block_ The packed function of `VisitBindingBlock_(const
   * BindingBlockNode* block)`.
   * \param f_visit_dataflow_block_ The packed function of `VisitBindingBlock_(const
   * DataflowBlockNode* block)`.
   * \param f_visit_var_def The packed function of `VisitVarDef(const Var& var)`.
   * \param f_visit_var_def_ The packed function of `VisitVarDef_(const VarNode* var)`.
   * \param f_visit_dataflow_var_def_ The packed function of `VisitVarDef_(const DataflowVarNode*
   * var)`.
   * \param f_visit_span The packed function of `VisitSpan(const Span& span)`.
   * \return The PyExprMutator created.
   */
  TVM_DLL static PyExprMutator MakePyExprMutator(
      BlockBuilder builder_, PackedFunc f_visit_expr, PackedFunc f_visit_constant_,
      PackedFunc f_visit_tuple_, PackedFunc f_visit_var_, PackedFunc f_visit_dataflow_var_,
      PackedFunc f_visit_shape_expr_, PackedFunc f_visit_extern_func_,
      PackedFunc f_visit_global_var_, PackedFunc f_visit_function_, PackedFunc f_visit_call_,
      PackedFunc f_visit_seq_expr_, PackedFunc f_visit_if_, PackedFunc f_visit_op_,
      PackedFunc f_visit_tuple_getitem_, PackedFunc f_visit_prim_value_,
      PackedFunc f_visit_string_imm_, PackedFunc f_visit_data_type_imm_, PackedFunc f_visit_binding,
      PackedFunc f_visit_var_binding_, PackedFunc f_visit_match_cast_,
      PackedFunc f_visit_binding_block, PackedFunc f_visit_binding_block_,
      PackedFunc f_visit_dataflow_block_, PackedFunc f_visit_var_def, PackedFunc f_visit_var_def_,
      PackedFunc f_visit_dataflow_var_def_, PackedFunc f_visit_span) {
    ObjectPtr<PyExprMutatorNode> n = make_object<PyExprMutatorNode>();
    n->builder_ = builder_;
    n->f_visit_expr = f_visit_expr;
    n->f_visit_constant_ = f_visit_constant_;
    n->f_visit_tuple_ = f_visit_tuple_;
    n->f_visit_var_ = f_visit_var_;
    n->f_visit_dataflow_var_ = f_visit_dataflow_var_;
    n->f_visit_shape_expr_ = f_visit_shape_expr_;
    n->f_visit_extern_func_ = f_visit_extern_func_;
    n->f_visit_global_var_ = f_visit_global_var_;
    n->f_visit_function_ = f_visit_function_;
    n->f_visit_call_ = f_visit_call_;
    n->f_visit_seq_expr_ = f_visit_seq_expr_;
    n->f_visit_if_ = f_visit_if_;
    n->f_visit_op_ = f_visit_op_;
    n->f_visit_tuple_getitem_ = f_visit_tuple_getitem_;
    n->f_visit_prim_value_ = f_visit_prim_value_;
    n->f_visit_string_imm_ = f_visit_string_imm_;
    n->f_visit_data_type_imm_ = f_visit_data_type_imm_;
    n->f_visit_binding = f_visit_binding;
    n->f_visit_var_binding_ = f_visit_var_binding_;
    n->f_visit_match_cast_ = f_visit_match_cast_;
    n->f_visit_binding_block = f_visit_binding_block;
    n->f_visit_binding_block_ = f_visit_binding_block_;
    n->f_visit_dataflow_block_ = f_visit_dataflow_block_;
    n->f_visit_var_def = f_visit_var_def;
    n->f_visit_var_def_ = f_visit_var_def_;
    n->f_visit_dataflow_var_def_ = f_visit_dataflow_var_def_;
    n->f_visit_span = f_visit_span;
    return PyExprMutator(n);
  }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyExprMutator, ObjectRef, PyExprMutatorNode);
};

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
