/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#include "tvm/relax/expr.h"

namespace tvm {
namespace relax {

using tvm::runtime::Optional;

TVM_REGISTER_NODE_TYPE(VarNode);

Var::Var(Id vid,
         Optional<Array<PrimExpr>> shape_annotation,
         Optional<Type> type_annotation,
         Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->shape_ = std::move(shape_annotation);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.Var")
.set_body_typed([](String name_hint,
                   Optional<Array<PrimExpr>> shape_annotation,
                   Optional<Type> type_annotation) {
  return Var(name_hint, shape_annotation, type_annotation);
});


TVM_REGISTER_NODE_TYPE(DataflowVarNode);

TVM_REGISTER_GLOBAL("relax.DataflowVar")
.set_body_typed([](String name_hint,
                   Optional<Array<PrimExpr>> shape_annotation,
                   Optional<Type> type_annotation) {
  return DataflowVar(name_hint, shape_annotation, type_annotation);
});


TVM_REGISTER_NODE_TYPE(BindingNode);

TVM_REGISTER_GLOBAL("relax.Binding")
.set_body_typed([]() {
  return Binding();
});


TVM_REGISTER_NODE_TYPE(MatchShapeNode);

MatchShape::MatchShape(Array<PrimExpr> pattern,
                       Expr value) {
  ObjectPtr<MatchShapeNode> n = make_object<MatchShapeNode>();
  n->pattern = std::move(pattern);
  n->value = std::move(value);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.MatchShape")
.set_body_typed([](Array<PrimExpr> pattern, Expr value) {
  return MatchShape(pattern, value);
});


TVM_REGISTER_NODE_TYPE(VarBindingNode);

VarBinding::VarBinding(Var var,
                       Expr value) {
  ObjectPtr<VarBindingNode> n = make_object<VarBindingNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.VarBinding")
.set_body_typed([](Var var,Expr value) {
  return VarBinding(var,value);
});


TVM_REGISTER_NODE_TYPE(BindingBlockNode);

BindingBlock::BindingBlock(Array<Binding> bindings) {
  ObjectPtr<BindingBlockNode> n = make_object<BindingBlockNode>();
  n->bindings = std::move(bindings);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.BindingBlock")
.set_body_typed([](Array<Binding> bindings) {
  return BindingBlock(bindings);
});


TVM_REGISTER_NODE_TYPE(DataflowBlockNode);

DataflowBlock::DataflowBlock(Array<Binding> bindings) {
  ObjectPtr<DataflowBlockNode> n = make_object<DataflowBlockNode>();
  n->bindings = std::move(bindings);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowBlock")
.set_body_typed([](Array<Binding> bindings) {
  return DataflowBlock(bindings);
});


TVM_REGISTER_NODE_TYPE(SeqExprNode);

SeqExpr::SeqExpr(Array<BindingBlock> blocks,
                 Expr body,
                 Type checked_type_,
                 Array<PrimExpr> shape_,
                 Span span) {
  ObjectPtr<SeqExprNode> n = make_object<SeqExprNode>();
  n->blocks = std::move(blocks);
  n->body = std::move(body);
  n->checked_type_ = std::move(checked_type_);
  n->shape_ = std::move(shape_);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.SeqExpr")
.set_body_typed([](Array<BindingBlock> blocks, Expr body, Type checked_type_, Array<PrimExpr> shape_, Span span) {
  return SeqExpr(blocks,body,checked_type_,shape_,span);
});


ShapeExpr::ShapeExpr(Array<PrimExpr> values,
                     Type checked_type_, Array<PrimExpr> shape_, Span span) {
    ObjectPtr<ShapeExprNode> n = make_object<ShapeExprNode>();
    n->values = std::move(values);
    n->checked_type_ = std::move(checked_type_);
    n->shape_ = std::move(shape_);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ShapeExprNode);

TVM_REGISTER_GLOBAL("relax.ShapeExpr").set_body_typed([](Array<PrimExpr> values, Type checked_type_,Array<PrimExpr> shape_,Span span) {
    return ShapeExpr(values, checked_type_,shape_, span);
});


Function::Function(
    runtime::Optional<GlobalVar> name,
    Array<Var> params,
    Expr body,
    Type ret_type,
    Type checked_type_,
    Array<PrimExpr> shape_,
    Span span) {
    ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
    n->name = std::move(name);
    n->params = std::move(params);
    n->body = std::move(body);
    n->ret_type = std::move(ret_type);
    n->checked_type_ = std::move(checked_type_);
    n->shape_ = std::move(shape_);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_GLOBAL("relax.Function").set_body_typed([](runtime::Optional<GlobalVar> name,Array<Var> params,Expr body,Type ret_type,Type checked_type_,Array<PrimExpr> shape_,Span span) {
    return Function(name,params,body,ret_type,checked_type_,shape_,span);
});

} // namespace relax
} // namespace tvm
