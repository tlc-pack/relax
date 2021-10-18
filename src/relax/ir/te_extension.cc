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
#include <tvm/relax/te_extension.h>

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

Array<IterVar> RXPlaceholderOpNode::root_iter_vars() const { return {}; }

DataType RXPlaceholderOpNode::output_dtype(size_t i) const {
  ICHECK_EQ(i, 0U);
  return dtype;
}

Array<PrimExpr> RXPlaceholderOpNode::output_shape(size_t i) const {
  ICHECK_EQ(i, 0U);
  return shape;
}

RXPlaceholderOp::RXPlaceholderOp(std::string name, Array<PrimExpr> shape, DataType dtype) {
  auto n = make_object<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->dtype = dtype;
  data_ = std::move(n);
}

Tensor placeholder(Array<PrimExpr> shape, DataType dtype, std::string name) {
  return PlaceholderOp(name, shape, dtype).output(0);
}

TVM_REGISTER_GLOBAL("te.Placeholder")
    .set_body_typed([](Array<PrimExpr> shape, DataType dtype, std::string name) {
      return placeholder(shape, dtype, name);
    });

Array<Tensor> PlaceholderOpNode::InputTensors() const { return {}; }

Operation PlaceholderOpNode::ReplaceInputs(const Operation& self,
                                           const std::unordered_map<Tensor, Tensor>& rmap) const {
  return self;
}

void PlaceholderOpNode::PropBoundToInputs(
    const Operation& self, arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {}

void PlaceholderOpNode::GatherBound(const Operation& self,
                                    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                                    std::unordered_map<IterVar, Range>* out_dom_map) const {}

Stmt PlaceholderOpNode::BuildRealize(const Stage& stage,
                                     const std::unordered_map<IterVar, Range>& realize_map,
                                     const Stmt& body) const {
  return body;
}

Stmt PlaceholderOpNode::BuildProvide(const Stage& stage,
                                     const std::unordered_map<IterVar, Range>& dom_map,
                                     bool debug_keep_trivial_loop) const {
  return Stmt();
}
}  // namespace relax
}  // namespace tvm
