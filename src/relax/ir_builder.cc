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
 * \file src/relax/ir_builder.cc
 */

#include <tvm/relax/ir_builder.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(IRBuilderNode);
TVM_REGISTER_NODE_TYPE(LazyIRBuilderNode);
TVM_REGISTER_NODE_TYPE(FunctionScopeNode);
TVM_REGISTER_NODE_TYPE(DataflowScopeNode);

IRBuilder IRBuilderNode::Create() {
  IRBuilder ret(make_object<IRBuilderNode>());
  return ret;
}

void IRBuilderNode::FillFuncNameParam(const Array<Var>& params, const std::string& func_name) {
  if (!func_name.empty()) {
    this->func_.func_name = GlobalVar(func_name);
  }
  for (Var param : params) {
    this->var_map_[param] = param;
  }
  this->func_.params = params;
}

void IRBuilderNode::BuildFunction() {
  SeqExpr seq = SeqExpr(this->func_.binding_blocks, this->func_.ret);
  this->func_.func = Function(this->func_.func_name, this->func_.params, seq, {});
  this->global_var_counter_ = 0;
}

void IRBuilderNode::BuildBlock() {
  if (!this->func_.bindings.empty()) {
    if (is_dataflow_) {
      this->func_.binding_blocks.emplace_back(DataflowBlock(this->func_.bindings));
    } else {
      this->func_.binding_blocks.emplace_back(BindingBlock(this->func_.bindings));
    }
    this->func_.bindings.clear();
  }
  this->dataflow_var_counter_ = 0;
  this->is_dataflow_ = !this->is_dataflow_;
}

Optional<RelayExpr> InferShape(const Call& call, DiagnosticContext diag_ctx) {
  auto op_map = Op::GetAttrMap<FInferShape>("FInferShape");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    return op_map[op](call, diag_ctx);
  } else {
    return NullOpt;
  } 
}

Type InferType(const Call& call, DiagnosticContext diag_ctx) {
  auto op_map = Op::GetAttrMap<FInferType>("FInferType");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    return op_map[op](call, diag_ctx);
  } else {
    return VoidType();
  }
}

Var IRBuilderNode::Emit(const Call& call) {
  Var var;
  if (is_dataflow_) {
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

  this->func_.bindings.emplace_back(VarBinding(var, call));
  this->var_map_[var] = call;
  return var;
}

Var IRBuilderNode::Emit(const VarBinding& binding) {
  this->func_.bindings.emplace_back(binding);
  this->var_map_[binding->var] = binding->value;
  return binding->var;
}

Var IRBuilderNode::Emit(const Var& var, const Call& call) {
  this->func_.bindings.emplace_back(VarBinding(var, call));
  this->var_map_[var] = call;
  return var;
}

Var IRBuilderNode::EmitOutput(const Expr& output) {
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

Function IRBuilderNode::Get() { return this->func_.func; }

std::vector<BindingBlock> IRBuilderNode::GetBlocks() { return this->func_.binding_blocks; }

class FunctionScope::Internal {
 public:
  static void ExitScope(FunctionScope scope) { scope.ExitWithScope(); }
};

FunctionScope::FunctionScope(IRBuilder ib) {
  ObjectPtr<FunctionScopeNode> n = make_object<FunctionScopeNode>();
  n->ir_builder = std::move(ib);
  data_ = std::move(n);
}

void FunctionScope::ExitWithScope() {
  this->get()->ir_builder->BuildBlock();
  this->get()->ir_builder->BuildFunction();
}

class DataflowScope::Internal {
 public:
  static void EnterScope(DataflowScope scope) { scope.EnterWithScope(); }

  static void ExitScope(DataflowScope scope) { scope.ExitWithScope(); }
};

DataflowScope::DataflowScope(IRBuilder ib) {
  ObjectPtr<DataflowScopeNode> n = make_object<DataflowScopeNode>();
  n->ir_builder = std::move(ib);
  data_ = std::move(n);
}

void DataflowScope::EnterWithScope() { this->get()->ir_builder->BuildBlock(); }

void DataflowScope::ExitWithScope() { this->get()->ir_builder->BuildBlock(); }

LazyIRBuilder LazyIRBuilderNode::Create(const DataflowBlock& block) {
  LazyIRBuilder ret(make_object<LazyIRBuilderNode>());
  ret->df_block_ = block;
  return ret;
}

Var LazyIRBuilderNode::Emit(const Call& call) {
  if (is_rewrite_) {
    index_++;
    return IRBuilderNode::Emit(call);
  }
  Expr expr = Downcast<VarBinding>(this->df_block_->bindings[index_])->value;
  Call old_call = Downcast<Call>(expr);
  if (call.same_as(old_call)) {
    VarBinding binding = Downcast<VarBinding>(this->df_block_->bindings[index_++]);
    this->var_map_[binding->var] = binding->value;
    return binding->var;
  }
  else {
    is_rewrite_ = true;
    for (int i = 0; i < index_; i++) {
      Expr expr = Downcast<VarBinding>(this->df_block_->bindings[i])->value;
      IRBuilderNode::Emit(Downcast<Call>(expr));
    }
    index_++;
    return IRBuilderNode::Emit(call);
  }
}

Var LazyIRBuilderNode::Emit(const VarBinding& binding) {
  if (is_rewrite_) {
    index_++;
    return IRBuilderNode::Emit(binding);
  }
  Binding old_binding = this->df_block_->bindings[index_];
  if (binding.same_as(old_binding)) {
    index_++;
    this->var_map_[binding->var] = binding->value;
    return binding->var;
  }
  else {
    is_rewrite_ = true;
    for (int i = 0; i < index_; i++) {
      Expr expr = Downcast<VarBinding>(this->df_block_->bindings[i])->value;
      IRBuilderNode::Emit(Downcast<Call>(expr));
    }
    index_++;
    Call call = Downcast<Call>(binding->value);
    return IRBuilderNode::Emit(call);
  }
}

Var LazyIRBuilderNode::Emit(const Var& var, const Call& call) {
  if (is_rewrite_) {
    index_++;
    return IRBuilderNode::Emit(var, call);
  }
  Expr expr = Downcast<VarBinding>(this->df_block_->bindings[index_])->value;
  Call old_call = Downcast<Call>(expr);
  if (call.same_as(old_call)) {
    index_++;
    this->var_map_[var] = call;
    return var;
  }
  else {
    is_rewrite_ = true;
    for (int i = 0; i < index_; i++) {
      Expr expr = Downcast<VarBinding>(this->df_block_->bindings[i])->value;
      IRBuilderNode::Emit(Downcast<Call>(expr));
    }
    index_++;
    return IRBuilderNode::Emit(var, call);
  }
}

void LazyIRBuilderNode::BuildBlock() {
  if (!this->func_.bindings.empty()) {
    if (is_dataflow_) {
      if (is_rewrite_) {
        this->func_.binding_blocks.emplace_back(DataflowBlock(this->func_.bindings));
      }
      else {
        this->func_.binding_blocks.emplace_back(this->df_block_);
      }
    } else {
      this->func_.binding_blocks.emplace_back(BindingBlock(this->func_.bindings));
    }
    this->func_.bindings.clear();
  }
  this->dataflow_var_counter_ = 0;
  this->is_dataflow_ = !this->is_dataflow_;
}

TVM_REGISTER_GLOBAL("relax.IRBuilderCreate").set_body_typed(IRBuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.IRBuilderFillFuncNameParam")
    .set_body_typed([](IRBuilder builder, const Array<Var>& params, const std::string& func_name) {
      return builder->FillFuncNameParam(params, func_name);
    });

TVM_REGISTER_GLOBAL("relax.IRBuilderBuildFunction").set_body_typed([](IRBuilder builder) {
  return builder->BuildFunction();
});

TVM_REGISTER_GLOBAL("relax.IRBuilderEmit").set_body_typed([](IRBuilder builder, const Call& call) {
  return builder->Emit(call);
});

TVM_REGISTER_GLOBAL("relax.IRBuilderEmitOutput")
    .set_body_typed([](IRBuilder builder, const Expr& output) {
      return builder->EmitOutput(output);
    });

TVM_REGISTER_GLOBAL("relax.IRBuilderGet").set_body_typed([](IRBuilder builder) {
  return builder->Get();
});

TVM_REGISTER_GLOBAL("relax.IRBuilderGetBlocks").set_body_typed([](IRBuilder builder) {
  return Array<BindingBlock>(builder->GetBlocks());
});

TVM_REGISTER_GLOBAL("relax.CreateFunctionScope").set_body_typed([](IRBuilder ib) {
  return FunctionScope(ib);
});

TVM_REGISTER_GLOBAL("relax.ExitFunctionScope").set_body_typed(FunctionScope::Internal::ExitScope);

TVM_REGISTER_GLOBAL("relax.CreateDataflowScope").set_body_typed([](IRBuilder ib) {
  return DataflowScope(ib);
});

TVM_REGISTER_GLOBAL("relax.EnterDataflowScope").set_body_typed(DataflowScope::Internal::EnterScope);

TVM_REGISTER_GLOBAL("relax.ExitDataflowScope").set_body_typed(DataflowScope::Internal::ExitScope);

}  // namespace relax
}  // namespace tvm
