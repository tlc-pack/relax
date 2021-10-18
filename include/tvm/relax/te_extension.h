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
 * \file tvm/relax/te_extension.h
 * \brief Tensor expression extension in Relax.
 */
#ifndef TVM_RELAX_TE_EXTENSION_H_
#define TVM_RELAX_TE_EXTENSION_H_

#include <tvm/relax/expr.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace relax {

class RXPlaceholderOpNode : public te::OperationNode {
  Expr value;

  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body, String storage_scope = "") const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    bool debug_keep_trivial_loop) const final;

  static constexpr const char* _type_key = "RXPlaceholderOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(RXPlaceholderOpNode, OperationNode);
};

class RXPlaceholderOp : public Operation {
 public:
  RXPlaceholderOp(Expr value);

  TVM_DEFINE_OBJECT_REF_METHODS(RXPlaceholderOp, Operation, RXPlaceholderOpNode);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TE_EXTENSION_H_
