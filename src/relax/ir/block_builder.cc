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
 * \file src/relax/block_builder.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relax {

// ================================
// BlockBuilderNode::ExprNormalizer

// TODO(@altanh): more test cases to cover different visits
class BlockBuilderNode::ExprNormalizer : public ExprFunctor<Expr(const Expr&)> {
 public:
  ExprNormalizer(BlockBuilderNode* builder) : builder_(builder) {}

#define RELAX_EXPR_NORMALIZER_LEAF(OP) \
  Expr VisitExpr_(const OP* op) final { return GetRef<Expr>(op); }

  RELAX_EXPR_NORMALIZER_LEAF(ConstantNode);
  RELAX_EXPR_NORMALIZER_LEAF(VarNode);
  RELAX_EXPR_NORMALIZER_LEAF(DataflowVarNode);
  RELAX_EXPR_NORMALIZER_LEAF(ShapeExprNode);
  RELAX_EXPR_NORMALIZER_LEAF(ExternFuncNode);
  RELAX_EXPR_NORMALIZER_LEAF(GlobalVarNode);
  RELAX_EXPR_NORMALIZER_LEAF(OpNode);

  // TODO(@altanh): CopyOnWrite

  Expr VisitExpr(const Expr& expr) {
    Optional<Expr> post = expr_memo_.Get(expr);
    if (post) {
      ICHECK(post.as<VarNode>()) << "memoized expressions should map to variables";
      return post.value();
    }
    return ExprFunctor::VisitExpr(expr);
  }

  Expr VisitExpr_(const TupleNode* op) final {
    bool unchanged = true;
    Array<Expr> new_fields;
    for (const Expr& field : op->fields) {
      Expr new_field = this->Bind(field);
      new_fields.push_back(new_field);
      unchanged &= new_field.same_as(field);
    }
    return unchanged ? GetRef<Expr>(op) : Tuple(new_fields);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Expr new_body = this->VisitWithNewScope(op->body);
    if (new_body.same_as(op->body)) {
      return GetRef<Expr>(op);
    }
    return Function(op->name, op->params, new_body, op->ret_type);
  }

  Expr VisitExpr_(const CallNode* op) final {
    Expr new_op = this->VisitExpr(op->op);
    bool unchanged = new_op.same_as(op->op);

    Array<Expr> new_args;
    for (const Expr& arg : op->args) {
      Expr new_arg = this->Bind(arg);
      new_args.push_back(new_arg);
      unchanged &= new_arg.same_as(arg);
    }

    if (unchanged) {
      return GetRef<Expr>(op);
    }
    return Call(new_op, new_args, op->attrs, op->type_args);
  }

  Expr VisitExpr_(const SeqExprNode* op) final {
    bool unchanged = true;
    Array<BindingBlock> new_blocks;
    for (const BindingBlock& block : op->blocks) {
      // TODO(@altanh): we could merge sequential non-dataflow BindingBlocks here
      BindingBlock new_block = this->VisitBindingBlock(block);
      new_blocks.push_back(new_block);
      unchanged &= new_block.same_as(block);
    }

    builder_->BeginBindingBlock();
    Expr new_body = this->VisitExpr(op->body);
    unchanged &= new_body.same_as(op->body);
    BindingBlock prologue = builder_->EndBlock();

    // TODO(@altanh, @yuchen): normalize nested SeqExprs and BindingBlocks

    if (!prologue->bindings.empty()) {
      new_blocks.push_back(prologue);
      unchanged = false;
    }

    if (unchanged) {
      return GetRef<Expr>(op);
    }
    return SeqExpr(new_blocks, new_body);
  }

  Expr VisitExpr_(const IfNode* op) final {
    Expr new_cond = this->VisitExpr(op->cond);
    Expr new_true = this->VisitWithNewScope(op->true_branch);
    Expr new_false = this->VisitWithNewScope(op->false_branch);
    if (new_cond.same_as(op->cond) && new_true.same_as(op->true_branch) &&
        new_false.same_as(op->false_branch)) {
      return GetRef<Expr>(op);
    }
    return If(new_cond, new_true, new_false);
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr new_tuple = this->VisitExpr(op->tuple);
    if (new_tuple.same_as(op->tuple)) {
      return GetRef<Expr>(op);
    }
    return TupleGetItem(new_tuple, op->index);
  }

  Binding VisitBinding(const Binding& binding) {
    if (binding.as<VarBindingNode>()) {
      return this->VisitVarBinding(Downcast<VarBinding>(binding));
    } else {
      ICHECK(binding.as<MatchShapeNode>()) << "expected VarBinding or MatchShape, got " << binding;
      return this->VisitMatchShape(Downcast<MatchShape>(binding));
    }
  }

  VarBinding VisitVarBinding(const VarBinding& binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (new_value.same_as(binding->value) || new_value.same_as(binding->var)) {
      // if new_value = binding->var, then we have found an ANF binding site, so just return it
      return binding;
    }
    return VarBinding(binding->var, new_value);
  }

  MatchShape VisitMatchShape(const MatchShape& binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (new_value.same_as(binding->value)) {
      return binding;
    }
    return MatchShape(new_value, binding->pattern, binding->var);
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) {
    if (block.as<DataflowBlockNode>()) {
      builder_->BeginDataflowBlock();
    } else {
      builder_->BeginBindingBlock();
    }

    bool unchanged = true;
    for (const Binding& binding : block->bindings) {
      Binding new_binding = this->VisitBinding(binding);
      unchanged &= new_binding.same_as(binding);
      if (new_binding.as<VarBindingNode>()) {
        VarBinding var_binding = Downcast<VarBinding>(new_binding);
        if (builder_->CurrentBlockIsDataFlow() && !var_binding->var.as<DataflowVarNode>()) {
          builder_->EmitOutput(var_binding);
        } else {
          builder_->Emit(var_binding);
        }
      } else {
        ICHECK(new_binding.as<MatchShapeNode>());
        builder_->EmitMatchShape(Downcast<MatchShape>(new_binding));
      }
    }
    BindingBlock new_block = builder_->EndBlock();
    unchanged &= new_block->bindings.size() == block->bindings.size();
    if (unchanged) {
      return block;
    }
    return new_block;
  }

 private:
  /*!
   * \brief Memoization map for expressions using Id for equality of variables.
   */
  class ExprMemo {
   public:
    Optional<Expr> Get(const Expr& expr) {
      if (const VarNode* var = expr.as<VarNode>()) {
        auto it = var_memo_.find(var->vid);
        if (it != var_memo_.end()) {
          return it->second;
        }
      } else {
        auto it = expr_memo_.find(expr);
        if (it != expr_memo_.end()) {
          return it->second;
        }
      }
      return NullOpt;
    }

    void Set(const Expr& pre, const Expr& post) {
      if (const VarNode* var = pre.as<VarNode>()) {
        var_memo_[var->vid] = post;
      } else {
        expr_memo_[pre] = post;
      }
    }

   private:
    std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual> var_memo_;
    std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> expr_memo_;
  };

  static bool IsLeaf(const Expr& expr) {
    // NB: tuples are treated as leaf nodes for ergonomics
    // TODO(@altanh, @yuchen): remove TupleNode from leaf
    return expr.as<VarNode>() || expr.as<GlobalVarNode>() || expr.as<relay::ConstantNode>() ||
           expr.as<ShapeExprNode>() || expr.as<ExternFuncNode>() || expr.as<OpNode>() ||
           expr.as<TupleNode>();
  }

  Expr VisitWithNewScope(const Expr& expr) {
    builder_->BeginBindingBlock();
    Expr post = this->VisitExpr(expr);
    BindingBlock prologue = builder_->EndBlock();
    if (!prologue->bindings.empty()) {
      post = SeqExpr({prologue}, post);
    }
    return post;
  }

  Expr Bind(const Expr& expr) {
    Expr post = this->VisitExpr(expr);
    if (!IsLeaf(post)) {
      post = builder_->Emit(post);
      expr_memo_.Set(expr, post);
    }
    return post;
  }

  /*! \brief BlockBuilder used for emitting intermediate variables. */
  BlockBuilderNode* builder_;

  /*! \brief Memoization table for mapping expressions to their ANF variables. */
  ExprMemo expr_memo_;
};

// ================
// BlockBuilderNode

TVM_REGISTER_NODE_TYPE(BlockBuilderNode);

BlockBuilderNode::BlockBuilderNode() {
  name_table_ = std::make_unique<NameTable>();
  normalizer_ = std::make_unique<ExprNormalizer>(this);
}

BlockBuilderNode::~BlockBuilderNode() {
  if (!block_stack_.empty()) {
    LOG(WARNING) << "BlockBuilder destroyed with remaining blocks!";
  }
}

void BlockBuilderNode::BeginDataflowBlock() { this->block_stack_.push({{}, true}); }

void BlockBuilderNode::BeginBindingBlock() { this->block_stack_.push({{}, false}); }

BindingBlock BlockBuilderNode::EndBlock() {
  BlockFrame* cur_frame = CurrentFrame();
  BindingBlock ret = cur_frame->is_dataflow ? DataflowBlock(cur_frame->bindings)
                                            : BindingBlock(cur_frame->bindings);
  block_stack_.pop();
  return ret;
}

Optional<Expr> InferShape(const Call& call, DiagnosticContext diag_ctx) {
  // if the call node's shape_ is filled, return the shape directly.
  if (call->shape_) {
    return Downcast<Expr>(call->shape_.value());
  }
  auto op_map = Op::GetAttrMap<FInferShape>("FInferShape");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    if (op_map.count(op)) {
      return op_map[op](call, diag_ctx);
    }
  }
  return NullOpt;
}

Type InferType(const Call& call, DiagnosticContext diag_ctx) {
  // if the call node's checked_type_ is filled, return the type directly.
  if (call->checked_type_.defined()) {
    return call->checked_type_;
  }
  auto op_map = Op::GetAttrMap<FInferType>("FInferType");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    if (op_map.count(op)) {
      return op_map[op](call, diag_ctx);
    }
  }
  return Type();
}

Var BlockBuilderNode::Emit(const Expr& expr, std::string name_hint) {
  return Emit(expr, CurrentFrame()->is_dataflow, name_hint);
}

Var BlockBuilderNode::Emit(const Expr& expr, bool is_dataflow, std::string name_hint) {
  BlockFrame* cur_frame = CurrentFrame();

  if (name_hint.empty()) {
    name_hint = is_dataflow ? "lv" : "gv";
  }
  Id vid = Id(name_table_->GetUniqueName(name_hint));
  Var var = is_dataflow ? DataflowVar(vid, NullOpt, NullOpt) : Var(vid, NullOpt, NullOpt);

  // do eager inference for Calls
  if (const CallNode* call_node = expr.as<CallNode>()) {
    // TypeInference::InferCall(...)
    const Call& call = GetRef<Call>(call_node);

    Optional<Expr> inferred_shape = InferShape(call, this->diag_ctx_);
    Type inferred_type = InferType(call, this->diag_ctx_);

    var->shape_ = inferred_shape;
    var->checked_type_ = inferred_type;

    Call new_call = Call(call->op, call->args, call->attrs, call->type_args, call->span);
    new_call->checked_type_ = inferred_type;
    new_call->shape_ = inferred_shape;

    cur_frame->bindings.push_back(VarBinding(var, new_call));
    this->binding_table_[var->vid] = new_call;
  } else if (const VarNode* var_node = expr.as<VarNode>()) {
    const Var& lhs_var = GetRef<Var>(var_node);
    if (lhs_var->shape_.defined()) {
      var->shape_ = lhs_var->shape_;
    }
    if (lhs_var->checked_type_.defined()) {
      var->checked_type_ = lhs_var->checked_type_;
    }
    cur_frame->bindings.push_back(VarBinding(var, lhs_var));
    this->binding_table_[var->vid] = lhs_var;
  } else {
    cur_frame->bindings.push_back(VarBinding(var, expr));
    binding_table_[var->vid] = expr;
  }

  return var;
}

Var BlockBuilderNode::Emit(const VarBinding& binding) {
  BlockFrame* cur_frame = CurrentFrame();
  if (cur_frame->is_dataflow) {
    ICHECK(binding->var.as<DataflowVarNode>())
        << "Emit can only be used for local bindings in a dataflow block, use EmitOutput for "
           "output bindings instead";
  }
  cur_frame->bindings.push_back(binding);
  binding_table_[binding->var->vid] = binding->value;
  return binding->var;
}

Var BlockBuilderNode::EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern,
                                     std::string name_hint) {
  BlockFrame* cur_frame = CurrentFrame();

  if (name_hint.empty()) {
    name_hint = cur_frame->is_dataflow ? "lv" : "gv";
  }
  Id vid = Id(name_table_->GetUniqueName(name_hint));
  Var var =
      cur_frame->is_dataflow ? DataflowVar(vid, NullOpt, NullOpt) : Var(vid, NullOpt, NullOpt);

  if (value->checked_type().as<ShapeTypeNode>()) {
    var->checked_type_ = ShapeType(Span());
  } else if (const DynTensorTypeNode* tty = value->checked_type().as<DynTensorTypeNode>()) {
    ShapeExpr shape = ShapeExpr(pattern);
    var->shape_ = shape;
    DataType dtype = tty->dtype;
    var->checked_type_ = DynTensorType(pattern.size(), dtype);
  } else {
    this->diag_ctx_.EmitFatal(
        Diagnostic::Error(value->span)
        << "The value passed to EmitMatchShape must be of DynTensorType or ShapeType.");
  }

  MatchShape match_shape = MatchShape(value, pattern, var);
  cur_frame->bindings.push_back(match_shape);
  return var;
}

Var BlockBuilderNode::EmitMatchShape(const MatchShape& binding) {
  BlockFrame* cur_frame = CurrentFrame();
  if (binding->var.defined()) {
    ICHECK(!cur_frame->is_dataflow || binding->var.as<DataflowVarNode>())
        << "EmitMatchShape can only be used for local bindings in a dataflow block.";
    ICHECK(cur_frame->is_dataflow || !binding->var.as<DataflowVarNode>())
        << "cannot emit dataflow vars outside a dataflow block: " << binding->var->name_hint();
  }
  cur_frame->bindings.push_back(binding);
  // TODO(@altanh, @yuchen): what value should we bind? Consider
  //    y = add(x, x)
  //    z = match_shape(y, (n, m))
  // We would like pass writers to match "z" with the "add" node but with extra shape info.
  // Maybe this logic could be deferred to a DFPattern-style rewriter?
  return binding->var;
}

Var BlockBuilderNode::EmitOutput(const Expr& output, std::string name_hint) {
  BlockFrame* cur_frame = CurrentFrame();

  ICHECK(cur_frame->is_dataflow) << "EmitOutput has to be called inside dataflow block.";

  return Emit(output, false, name_hint);
}

Var BlockBuilderNode::EmitOutput(const VarBinding& binding) {
  BlockFrame* cur_frame = CurrentFrame();

  ICHECK(cur_frame->is_dataflow) << "EmitOutput has to be called inside dataflow block.";
  ICHECK(!binding->var.as<DataflowVarNode>()) << "EmitOutput can only emit Var bindings.";

  cur_frame->bindings.push_back(binding);
  binding_table_[binding->var->vid] = binding->value;
  return binding->var;
}

Expr BlockBuilderNode::LookupBinding(const Var& var) {
  auto it = binding_table_.find(var->vid);
  if (it == binding_table_.end()) {
    this->diag_ctx_.EmitFatal(Diagnostic::Error(var->span)
                              << "The var to be looked up is not in the binding table.");
  }
  return it->second;
}

bool BlockBuilderNode::CanProveShapeEqual(const Expr& lhs, const Expr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  const auto* lhs_shape = lhs.as<ShapeExprNode>();
  const auto* rhs_shape = rhs.as<ShapeExprNode>();
  if (lhs_shape && rhs_shape) {
    size_t lhs_ndim = lhs_shape->values.size();
    size_t rhs_ndim = rhs_shape->values.size();
    if (lhs_ndim != rhs_ndim) {
      return false;
    }
    arith::Analyzer analyzer;
    for (size_t i = 0; i < lhs_ndim; ++i) {
      PrimExpr lhs_dim = lhs_shape->values[i];
      PrimExpr rhs_dim = rhs_shape->values[i];
      if (!analyzer.CanProveEqual(lhs_dim, rhs_dim)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// TODO(@altanh, @yuchen): need an internal Emit_ that doesn't call normalize
Expr BlockBuilderNode::Normalize(const Expr& expr) {
  // TODO(@altanh): fast path
  Expr normalized = normalizer_->VisitExpr(expr);
  if (normalized.as<CallNode>()) {
    // FIXME(@altanh): potentially breaks idempotency
    Call call = Downcast<Call>(normalized);

    // only do shape/type inference if the call does not have shape/type
    if (call->shape_ && call->checked_type_.defined()) {
      return call;
    }

    // Shape inference
    if (!call->shape_) {
      auto inferred_shape = InferShape(call, this->diag_ctx_);
      if (inferred_shape) {
        call->shape_ = this->Normalize(inferred_shape.value());
      }
    }

    if (!call->checked_type_.defined()) {
      // Type inference
      auto inferred_type = InferType(call, this->diag_ctx_);
      call->checked_type_ = inferred_type;
    }

    return call;
  }
  return normalized;
}

BlockBuilderNode::BlockFrame* BlockBuilderNode::CurrentFrame() {
  ICHECK(!block_stack_.empty()) << "no block is being built";
  return &block_stack_.top();
}

NameTable* BlockBuilderNode::name_table() { return name_table_.get(); }

BlockBuilder BlockBuilder::Create() { return BlockBuilder(make_object<BlockBuilderNode>()); }

TVM_REGISTER_GLOBAL("relax.BlockBuilderCreate").set_body_typed(BlockBuilder::Create);

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginDataflowBlock")
    .set_body_typed([](BlockBuilder builder) { builder->BeginDataflowBlock(); });

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginBindingBlock").set_body_typed([](BlockBuilder builder) {
  builder->BeginBindingBlock();
});

TVM_REGISTER_GLOBAL("relax.BlockBuilderEndBlock").set_body_typed([](BlockBuilder builder) {
  return builder->EndBlock();
});

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmit")
    .set_body_typed([](BlockBuilder builder, const Call& call) { return builder->Emit(call); });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitMatchShape")
    .set_body_typed([](BlockBuilder builder, const Expr& value, const Array<PrimExpr>& pattern) {
      return builder->EmitMatchShape(value, pattern);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitOutput")
    .set_body_typed([](BlockBuilder builder, const Expr& output) {
      return builder->EmitOutput(output);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderNormalize")
    .set_body_typed([](BlockBuilder builder, const Expr& expr) {
      return builder->Normalize(expr);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderGetUniqueName")
    .set_body_typed([](BlockBuilder builder, String name_hint) {
      return builder->name_table()->GetUniqueName(name_hint);
    });

}  // namespace relax
}  // namespace tvm
