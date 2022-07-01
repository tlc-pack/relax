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

#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {
class Var2ValAnalysis : public relax::ExprVisitor {
 public:
  tvm::runtime::Map<Var, Expr> var2value_;
  void VisitBinding_(const VarBindingNode* binding) override {
    var2value_.Set(binding->var, binding->value);
  }
};

tvm::runtime::Map<Var, Expr> AnalyzeVar2Value(const Expr& expr) {
  Var2ValAnalysis var2val_analysis;
  var2val_analysis.VisitExpr(expr);
  return std::move(var2val_analysis.var2value_);
}

tvm::runtime::Map<Var, Expr> AnalyzeVar2Value(const IRModule& m) {
  Var2ValAnalysis var2val_analysis;

  for (const auto& it : m->functions) {
    // visit relax.Function
    if (auto* n = it.second.as<FunctionNode>()) {
      var2val_analysis.VisitExpr(GetRef<Function>(n));
    }
  }

  return std::move(var2val_analysis.var2value_);
}

TVM_REGISTER_GLOBAL(("relax.analysis.get_var2val")).set_body_typed([](const Function& f) {
  return AnalyzeVar2Value(f);
});

}  // namespace relax
}  // namespace tvm
