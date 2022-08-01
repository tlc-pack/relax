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
 * \file src/relax/analysis/udchain.cc
 * \brief Implementation of use-def analysis.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/op.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

class UDChain : public relax::ExprVisitor {
 public:
  std::map<const VarNode*, std::set<const VarNode*>> def2use;

  const VarNode* cur_user_;

  void VisitBinding_(const VarBindingNode* binding) override {
    // init
    cur_user_ = binding->var.get();
    this->VisitVarDef(binding->var);
    this->VisitExpr(binding->value);
    cur_user_ = nullptr;
  }

  void VisitExpr_(const VarNode* op) override {
    if (nullptr == cur_user_) return;

    def2use[op].insert(cur_user_);
  }
  void VisitVarDef(const Var& var) override { def2use[var.get()] = {}; }

  void VisitExpr_(const DataflowVarNode* op) override {
    VisitExpr_(static_cast<const VarNode*>(op));
  }
};

runtime::Map<Var, Array<Var>> UseDefChain(const DataflowBlock& dfb) {
  UDChain udchain;
  udchain.VisitBindingBlock_(dfb.get());
  runtime::Map<Var, Array<Var>> ret;
  for (const auto& kv : udchain.def2use) {
    Array<Var> uses{};
    uses.reserve(kv.second.size());
    for (const auto& v : kv.second) uses.push_back(GetRef<Var>(v));
    ret.Set(GetRef<Var>(kv.first), std::move(uses));
  }
  return ret;
}

TVM_REGISTER_GLOBAL("relax.analysis.udchain").set_body_typed(UseDefChain);

}  // namespace relax
}  // namespace tvm
