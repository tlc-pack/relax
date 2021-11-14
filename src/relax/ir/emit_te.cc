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
 * \file relax/src/ir/te_extension.cc
 * \brief 
 */
#include "./emit_te.h"

namespace tvm {
namespace relax {

// RXPlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RXPlaceholderOpNode>([](const ObjectRef& node, ReprPrinter* p) {
  auto* op = static_cast<const RXPlaceholderOpNode*>(node.get());
  p->stream << "rxplaceholder(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(RXPlaceholderOpNode);

int RXPlaceholderOpNode::num_outputs() const { return 1; }

Array<tir::IterVar> RXPlaceholderOpNode::root_iter_vars() const { return {}; }

DataType RXPlaceholderOpNode::output_dtype(size_t i) const {
  ICHECK_EQ(i, 0U);
  return DataType::Float(32);
}

Array<PrimExpr> RXPlaceholderOpNode::output_shape(size_t i) const {
  ICHECK_EQ(i, 0U);
  return Downcast<ShapeExpr>(value->shape())->values;
}

RXPlaceholderOp::RXPlaceholderOp(std::string name, Expr value) {
  auto n = make_object<RXPlaceholderOpNode>();
  n->name = name;
  n->value = value;
  data_ = std::move(n);
}

te::Tensor rxplaceholder(Expr value, std::string name) {
  return RXPlaceholderOp(name, value).output(0);
}

TVM_REGISTER_GLOBAL("relax.RXPlaceholder")
.set_body_typed([](Expr value, std::string name) {
  return rxplaceholder(value, name);
});

Array<te::Tensor> RXPlaceholderOpNode::InputTensors() const {
  return {};
}

te::Operation RXPlaceholderOpNode::ReplaceInputs(
    const te::Operation& self,
    const std::unordered_map<te::Tensor, te::Tensor>& rmap) const {
  return self;
}

void RXPlaceholderOpNode::PropBoundToInputs(
    const te::Operation& self, arith::Analyzer* analyzer,
    const std::unordered_map<const tir::VarNode*, arith::IntSet>& dom_map,
    std::unordered_map<te::Tensor, te::TensorDom>* out_dom_map) const {}

void RXPlaceholderOpNode::GatherBound(
    const te::Operation& self,
    const std::unordered_map<te::Tensor, te::TensorDom>& tensor_dom,
    std::unordered_map<tir::IterVar, Range>* out_dom_map) const {}

tir::Stmt RXPlaceholderOpNode::BuildRealize(
    const te::Stage& stage,
    const std::unordered_map<tir::IterVar, Range>& realize_map,
    const tir::Stmt& body,
    String storage_scope) const {
  return body;
}

tir::Stmt RXPlaceholderOpNode::BuildProvide(
    const te::Stage& stage,
    const std::unordered_map<tir::IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  return tir::Stmt();
}
}  // namespace relax
}  // namespace tvm
