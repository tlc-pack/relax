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

#include <tvm/node/node.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>

namespace tvm {
namespace relax {

class UseDefAnalysis : public relax::ExprVisitor {
 public:
  std::map<VarBinding, std::set<VarBinding>> udmap_;
  std::map<const VarNode*, const VarBindingNode*> v2binding_;
  VarBindingNode const* cur_vbinding_;

  void VisitBinding_(const VarBindingNode* binding) override {
    v2binding_[binding->var.get()] = cur_vbinding_ = binding;
    this->VisitExpr(binding->value);
    this->VisitVarDef(binding->var);
    cur_vbinding_ = nullptr;
  }

  void VisitExpr_(const VarNode* op) override {
    if (nullptr == cur_vbinding_) return;

    auto it = v2binding_.find(op);
    if (it != v2binding_.end()) {
      // it->second used by cur_vbinding_;
      udmap_[GetRef<VarBinding>(it->second)].insert(GetRef<VarBinding>(cur_vbinding_));
    }
  }

  void VisitExpr_(const DataflowVarNode* op) override {
    VisitExpr_(static_cast<const VarNode*>(op));
  }
};

std::map<VarBinding, std::set<VarBinding>> AnalyzeUDChain(const IRModule& m) {
  UseDefAnalysis udanalysis;

  for (const auto& it : m->functions) {
    // visit relax.Function
    if (auto* n = it.second.as<FunctionNode>()) {
      udanalysis.VisitExpr(GetRef<Function>(n));
    }
  }

  return std::move(udanalysis.udmap_);
}

TVM_REGISTER_GLOBAL(("relax.analysis.udchain")).set_body_typed([](IRModule m) {
  runtime::Map<VarBinding, Array<VarBinding>> ret;
  auto std_map = AnalyzeUDChain(m);
  for (const auto& it : std_map) {
    ret.Set(it.first, Array<VarBinding>(it.second.begin(), it.second.end()));
  }
  return ret;
});

}  // namespace relax
}  // namespace tvm