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
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(BlockBuilderNode);

BlockBuilder BlockBuilderNode::Create() {
  BlockBuilder ret(make_object<BlockBuilderNode>());
  return ret;
}

void BlockBuilderNode::BeginBlock(bool is_dataflow) {
  this->dataflow_var_counter_ = 0;
  this->block_stack_.push({{}, is_dataflow});
}

BindingBlock BlockBuilderNode::EndBlock() {
  Array<Binding> bindings = block_stack_.top().bindings;
  bool is_df = block_stack_.top().is_dataflow;
  BindingBlock ret;
  if (is_df) {
    ret = DataflowBlock(bindings);
  } else {
    ret = BindingBlock(bindings);
  }
  block_stack_.pop();
  return ret;
}

Optional<RelayExpr> InferShape(const Call& call, DiagnosticContext diag_ctx) {
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
  auto op_map = Op::GetAttrMap<FInferType>("FInferType");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    if (op_map.count(op)) {
      return op_map[op](call, diag_ctx);
    }
  }
  return VoidType();
}

Var BlockBuilderNode::Emit(const Call& call) {
  Var var;
  if (block_stack_.top().is_dataflow) {
    var = DataflowVar(Id("lv" + std::to_string(dataflow_var_counter_++)), NullOpt, NullOpt);
  } else {
    var = Var(Id("gv" + std::to_string(global_var_counter_++)), NullOpt, NullOpt);
  }

  // Shape inference
  auto inferred_shape = InferShape(call, this->diag_ctx_);
  if (inferred_shape.defined()) {
    if (auto* shape_expr = inferred_shape.value().as<ShapeExprNode>()) {
      call->shape_ = GetRef<Expr>(shape_expr);
      var->shape_ = call->shape_;
    }
  }
  // Type inference
  auto inferred_type = InferType(call, this->diag_ctx_);
  call->checked_type_ = inferred_type;
  var->checked_type_ = inferred_type;

  this->block_stack_.top().bindings.push_back(VarBinding(var, call));
  this->var_map_[var] = call;
  return var;
}

Var BlockBuilderNode::EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern) {
  Var var;
  if (block_stack_.top().is_dataflow) {
    var = DataflowVar(Id("lv" + std::to_string(dataflow_var_counter_++)), NullOpt, NullOpt);
  } else {
    var = Var(Id("gv" + std::to_string(global_var_counter_++)), NullOpt, NullOpt);
  }
  if (value->checked_type().as<ShapeTypeNode>()) {
    var->checked_type_ = ShapeType(Span());
  } else if (value->checked_type().as<DynTensorTypeNode>()) {
    ShapeExpr shape = ShapeExpr(pattern);
    var->shape_ = shape;
    DataType dtype = (Downcast<DynTensorType>(value->checked_type()))->dtype;
    var->checked_type_ = DynTensorType(pattern.size(), dtype);
  } else {
    this->diag_ctx_.EmitFatal(
        Diagnostic::Error(value->span)
        << "The value passed to EmitMatchShape must be of DynTensorType or ShapeType.");
  }

  MatchShape match_shape = MatchShape(value, pattern, var);
  this->block_stack_.top().bindings.push_back(match_shape);
  return var;
}

Var BlockBuilderNode::Emit(const VarBinding& binding) {
  // FIXME(yuchen or ziheng): consider binding in normal block)
  if (!binding->var.as<DataflowVarNode>()) {
    return EmitOutput(binding->var, binding->value);
  } else {
    this->block_stack_.top().bindings.push_back(binding);
    this->var_map_[binding->var] = binding->value;
    return binding->var;
  }
}

Var BlockBuilderNode::Emit(const Var& var, const Call& call) {
  Expr normalized_call = Normalize(call);
  // Reuse the input var if the shape and type of the call matches the var
  if (CanProveShapeEqual(var->shape(), call->shape()) &&
      StructuralEqual()(var->checked_type(), call->checked_type())) {
    this->block_stack_.top().bindings.push_back(VarBinding(var, normalized_call));
    this->var_map_[var] = normalized_call;
    return var;
  } else {
    Var new_var;
    if (normalized_call->shape_.defined()) {
      new_var->shape_ = normalized_call->shape_;
    }
    this->block_stack_.top().bindings.push_back(VarBinding(new_var, normalized_call));
    this->var_map_[new_var] = normalized_call;
    return new_var;
  }
}

Var BlockBuilderNode::EmitOutput(const Var& var, const Expr& output) {
  if (block_stack_.top().is_dataflow) {
    // Reuse the input var if the shape and type of the call matches the var
    if (CanProveShapeEqual(var->shape(), output->shape()) &&
        StructuralEqual()(var->checked_type(), output->checked_type())) {
      this->block_stack_.top().bindings.push_back(VarBinding(var, output));
      this->var_map_[var] = output;
      return var;
    } else {
      Var ret = Var(Id("gv" + std::to_string(global_var_counter_++)), NullOpt, NullOpt);
      ret->shape_ = output->shape_;
      ret->checked_type_ = output->checked_type_;
      this->block_stack_.top().bindings.push_back(VarBinding(ret, output));
      this->var_map_[ret] = output;
      return ret;
    }
  } else {
    this->diag_ctx_.EmitFatal(Diagnostic::Error(var->span)
                              << "EmitOutput has to be called inside dataflow block.");
  }
}

Var BlockBuilderNode::EmitOutput(const Expr& output) {
  Var ret;
  if (block_stack_.top().is_dataflow) {
    // Reuse the input var if the shape and type of the call matches the var
    ret = Var(Id("gv" + std::to_string(global_var_counter_++)), NullOpt, NullOpt);
    ret->shape_ = output->shape_;
    ret->checked_type_ = output->checked_type_;
    this->block_stack_.top().bindings.push_back(VarBinding(ret, output));
    this->var_map_[ret] = output;
  } else {
    this->diag_ctx_.EmitFatal(Diagnostic::Error(output->span)
                              << "EmitOutput has to be called inside dataflow block.");
  }
  return ret;
}

Expr BlockBuilderNode::LookupVar(const Var& var) {
  auto it = this->var_map_.find(var);
  if (it == this->var_map_.end()) {
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

Expr BlockBuilderNode::Normalize(const Expr& expr) {
  if (expr.as<CallNode>()) {
    Call call = Downcast<Call>(expr);
    // Shape inference
    auto inferred_shape = InferShape(call, this->diag_ctx_);
    if (inferred_shape.defined()) {
      if (auto* shape_expr = inferred_shape.value().as<ShapeExprNode>()) {
        call->shape_ = GetRef<Expr>(shape_expr);
      }
    }
    // Type inference
    auto inferred_type = InferType(call, this->diag_ctx_);
    call->checked_type_ = inferred_type;
    return call;
  }
  return expr;
}

TVM_REGISTER_GLOBAL("relax.BlockBuilderCreate").set_body_typed(BlockBuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginBlock")
    .set_body_typed([](BlockBuilder builder, bool is_dataflow) {
      builder->BeginBlock(is_dataflow);
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

// TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitVarOutput")
//     .set_body_typed([](BlockBuilder builder, const Var& var, const Expr& output) {
//       return builder->EmitOutput(var, output);
//     });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitOutput")
    .set_body_typed([](BlockBuilder builder, const Expr& output) {
      return builder->EmitOutput(output);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderNormalize")
    .set_body_typed([](BlockBuilder builder, const Expr& expr) {
      return builder->Normalize(expr);
    });

}  // namespace relax
}  // namespace tvm
