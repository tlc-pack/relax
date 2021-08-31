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
#include <tvm/relay/op.h>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(IRBuilderNode);
TVM_REGISTER_NODE_TYPE(FunctionScopeNode);
TVM_REGISTER_NODE_TYPE(DataflowScopeNode);

IRBuilder IRBuilderNode::Create() {
  IRBuilder ret(make_object<IRBuilderNode>());
  return ret;
}

void IRBuilderNode::FillFuncNameParam(const Array<Var>& params, const std::string& func_name) {
  if (!func_name.empty()) {
    this->func.func_name = GlobalVar(func_name);
  }
  
  this->func.params = params;
}

void IRBuilderNode::BuildFunction() {
  SeqExpr seq = SeqExpr(this->func.binding_blocks, this->func.ret);
  this->func.func = Function(this->func.func_name, this->func.params, seq, {});
  this->global_var_counter = 0;
}

void IRBuilderNode::BuildBlock() {
  if (!this->func.bindings.empty()) {
    if (is_dataflow) {
      this->func.binding_blocks.emplace_back(DataflowBlock(this->func.bindings));
    } else {
      this->func.binding_blocks.emplace_back(BindingBlock(this->func.bindings));
    }
    this->func.bindings.clear();
  }
  this->dataflow_var_counter = 0;
  this->is_dataflow = !this->is_dataflow;
}

Var IRBuilderNode::Emit(const Call& call) {
  Var var;
  if (is_dataflow) {
    var = DataflowVar(Id("lv" + std::to_string(dataflow_var_counter++)), NullOpt, NullOpt);
  } else {
    var = Var(Id("gv" + std::to_string(global_var_counter++)), NullOpt, NullOpt);
  }

  this->func.bindings.emplace_back(VarBinding(var, call));
  return var;
}

Var IRBuilderNode::EmitOutput(const Expr& output) {
  Var ret;
  if (is_dataflow) {
    ret = Var(Id("gv" + std::to_string(global_var_counter++)), NullOpt, NullOpt);
    ret->shape_ = output->shape_;
    ret->checked_type_ = output->checked_type_;
    this->func.bindings.emplace_back(VarBinding(ret, output));
  } else {
    this->func.ret = output;
  }
  return ret;
}

Function IRBuilderNode::Get() { return this->func.func; }

std::vector<BindingBlock> IRBuilderNode::GetBlocks() { return this->func.binding_blocks; }

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

void DataflowScope::EnterWithScope() {
  this->get()->ir_builder->BuildBlock();
}

void DataflowScope::ExitWithScope() {
  this->get()->ir_builder->BuildBlock();
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
