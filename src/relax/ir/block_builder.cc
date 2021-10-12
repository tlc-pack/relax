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

#include <tvm/relax/block_builder.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/op.h>
#include <tvm/arith/analyzer.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(IRBuilderNode);
// TVM_REGISTER_NODE_TYPE(LazyIRBuilderNode);
// TVM_REGISTER_NODE_TYPE(FunctionScopeNode);
// TVM_REGISTER_NODE_TYPE(DataflowScopeNode);

IRBuilder IRBuilderNode::Create() {
  IRBuilder ret(make_object<IRBuilderNode>());
  return ret;
}

void IRBuilderNode::BeginBlock(bool is_dataflow) {
  block_stack_.push({{}, is_dataflow});
}

BindingBlock IRBuilderNode::EndBlock() {
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

Var IRBuilderNode::Emit(const Call& call) {
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

  this->var_map_[var] = call;
  return var;
}

Var IRBuilderNode::EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern) {
  Var var;
  if (block_stack_.top().is_dataflow) {
    var = DataflowVar(Id("lv" + std::to_string(dataflow_var_counter_++)), NullOpt, NullOpt);
  } else {
    var = Var(Id("gv" + std::to_string(global_var_counter_++)), NullOpt, NullOpt);
  }
  if (value->checked_type().as<ShapeTypeNode>()) {
    var->checked_type_ = ShapeType(Span());
  } else if (value->checked_type().as<DynTensorTypeNode>()){
    ShapeExpr shape = ShapeExpr(pattern);
    var->shape_ = shape;
    DataType dtype = (Downcast<DynTensorType>(value->checked_type()))->dtype;
    var->checked_type_ = DynTensorType(pattern.size(), dtype);
  } else {
    this->diag_ctx_.EmitFatal(Diagnostic::Error(value->span) 
                              << "The value passed to EmitMatchShape must be of DynTensorType or ShapeType.");
  }

  MatchShape match_shape = MatchShape(value, pattern, var);
  return var;
}

Var IRBuilderNode::Emit(const VarBinding& binding) {
  // FIXME(yuchen or ziheng): consider binding in normal block)
  if (!binding->var.as<DataflowVarNode>()) {
    return EmitOutput(binding->value);
  } else {
    this->func_.bindings.emplace_back(binding);
    this->var_map_[binding->var] = binding->value;
    return binding->var;
  }
}

Var IRBuilderNode::Emit(const Var& var, const Call& call) {
  Expr normalized_call = Normalize(call);
  // Reuse the input var if the shape and type of the call matches the var
  if (CanProveShapeEqual(var->shape(), call->shape()) && StructuralEqual()(var->checked_type(), call->checked_type())) { 
    this->func_.bindings.emplace_back(VarBinding(var, normalized_call));
    this->var_map_[var] = normalized_call;
    return var;
  } else {
    Var new_var;
    if (normalized_call->shape_.defined()) {
      new_var->shape_ = normalized_call->shape_;
    }
    this->func_.bindings.emplace_back(VarBinding(new_var, normalized_call));
    this->var_map_[new_var] = normalized_call;
    return new_var;
  }
}

Var IRBuilderNode::EmitOutput(const Var& var, const Expr& output) {
  Var ret;
  if (is_dataflow_) {
    ret = Var(Id("gv" + std::to_string(global_var_counter_++)), NullOpt, NullOpt);
    ret->shape_ = output->shape_;
    ret->checked_type_ = output->checked_type_;
    this->func_.bindings.emplace_back(VarBinding(ret, output));
    this->var_map_[ret] = output;
  } else {
    this->func_.ret = output;
  }
  return ret;
}

Expr IRBuilderNode::LookupVar(const Var& var) {
  auto it = this->var_map_.find(var);
  if (it == this->var_map_.end()) {
    this->diag_ctx_.EmitFatal(Diagnostic::Error(var->span) 
                              << "The var to be looked up is not in the binding table.");
  }
  return it->second;
}

bool IRBuilderNode::CanProveShapeEqual(const Expr& lhs, const Expr& rhs) {
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

Expr IRBuilderNode::Normalize(const Expr& expr) {
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

TVM_REGISTER_GLOBAL("relax.IRBuilderCreate").set_body_typed(IRBuilderNode::Create);


TVM_REGISTER_GLOBAL("relax.IRBuilderEmit").set_body_typed([](IRBuilder builder, const Call& call) {
  return builder->Emit(call);
});

TVM_REGISTER_GLOBAL("relax.IRBuilderEmitMatchShape").set_body_typed([](IRBuilder builder, const Expr& value, const Array<PrimExpr>& pattern) {
  return builder->EmitMatchShape(value, pattern);
});

TVM_REGISTER_GLOBAL("relax.IRBuilderEmitOutput")
    .set_body_typed([](IRBuilder builder, const Var& var, const Expr& output) {
      return builder->EmitOutput(var, output);
    });

TVM_REGISTER_GLOBAL("relax.IRBuilderNormalize")
    .set_body_typed([](IRBuilder builder, const Expr& expr) {
      return builder->Normalize(expr);
    });

}  // namespace relax
}  // namespace tvm
