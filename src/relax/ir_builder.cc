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
#include <tvm/relax/op_attr.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relax {

using relay::Call;

TVM_REGISTER_NODE_TYPE(IRBuilderNode);

IRBuilder IRBuilderNode::Create() {
  IRBuilder ret(make_object<IRBuilderNode>());
  return ret;
}

void IRBuilderNode::BuildFunction(std::string func_name, Array<Var> params) {
  SeqExpr seq = SeqExpr(this->func.binding_blocks, this->func.ret);
  if (func.ret.defined()) {
    this->func.func = Function(GlobalVar(func_name), params, seq, this->func.ret->checked_type_);
  } else {
    this->func.func = Function(GlobalVar(func_name), params, seq, {});
  }
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
}

Optional<RelayExpr> InferShape(Call call) {
  auto op_map = Op::GetAttrMap<relax::FInferShape>("FInferShape");
  if (const auto* op_node = call->op.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    if (op_map.count(op)) {
      return op_map[op](call);
    }
  }
  return NullOpt;
}

Type InferType(Call call) {
  auto op_map = Op::GetAttrMap<relax::FInferType>("FInferType");
  if (const auto* op_node = call->op.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    if (op_map.count(op)) {
      return op_map[op](call);
    }
  }
  return VoidType();
}

Var IRBuilderNode::Emit(Call call) {
  Var var;
  if (is_dataflow) {
    var = DataflowVar(Id("lv" + std::to_string(dataflow_var_counter++)), NullOpt, NullOpt);
  } else {
    var = Var(Id("gv" + std::to_string(global_var_counter++)), NullOpt, NullOpt);
  }
  // Shape inference
  auto inferred_shape = InferShape(call);
  if (inferred_shape) {
    if (auto* shape_expr = inferred_shape.value().as<ShapeExprNode>()) {
      call->shape_ = shape_expr->values;
      var->shape_ = shape_expr->values;
    }
  }
  // Type inference
  auto inferred_type = InferType(call);
  call->checked_type_ = inferred_type;
  var->checked_type_ = inferred_type;

  this->func.bindings.emplace_back(VarBinding(var, call));
  return var;
}

Var IRBuilderNode::EmitDataflowOutput(Var var) {
  Var ret;
  if (is_dataflow) {
    ret = Var(Id("gv" + std::to_string(global_var_counter++)), NullOpt, NullOpt);
    ret->shape_ = var->shape_;
    ret->checked_type_ = var->checked_type_;
    this->func.bindings.emplace_back(VarBinding(ret, var));
  } else {
    LOG(FATAL) << "EmitDataflowOutput must be called inside a dataflow block";
  }
  return ret;
}

void IRBuilderNode::EmitOutput(Expr output) { this->func.ret = output; }

inline void IRBuilderNode::SwitchBlock() { is_dataflow = !is_dataflow; }

Function IRBuilderNode::Get() { return this->func.func; }

TVM_REGISTER_GLOBAL("relax.IRBuilderCreate").set_body_typed(IRBuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.IRBuilderBuildBlock").set_body_typed([](IRBuilder builder) {
  return builder->BuildBlock();
});

TVM_REGISTER_GLOBAL("relax.IRBuilderBuildFunction")
    .set_body_typed([](IRBuilder builder, std::string func_name, Array<Var> params) {
      return builder->BuildFunction(func_name, params);
    });

TVM_REGISTER_GLOBAL("relax.IRBuilderEmit").set_body_typed([](IRBuilder builder, Call call) {
  return builder->Emit(call);
});

TVM_REGISTER_GLOBAL("relax.IRBuilderEmitDataflowOutput")
    .set_body_typed([](IRBuilder builder, Var var) { return builder->EmitDataflowOutput(var); });

TVM_REGISTER_GLOBAL("relax.IRBuilderEmitOutput").set_body_typed([](IRBuilder builder, Expr output) {
  builder->EmitOutput(output);
});

TVM_REGISTER_GLOBAL("relax.IRBuilderGet").set_body_typed([](IRBuilder builder) {
  return builder->Get();
});

TVM_REGISTER_GLOBAL("relax.IRBuilderSwitchBlock").set_body_typed([](IRBuilder builder) {
  return builder->SwitchBlock();
});

}  // namespace relax
}  // namespace tvm
