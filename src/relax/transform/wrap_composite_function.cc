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

#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include "utils.h"

namespace tvm {
namespace relax {

class WrapComposite : public ExprMutator {
 public:
  explicit WrapComposite(IRModule mod) : ExprMutator(mod) {}
  using ExprMutator::VisitExpr_;

  IRModule Run() {
    auto mod = builder_->GetContextIRModule();
    auto gvar = mod->GetGlobalVar("main");
    auto func = Downcast<Function>(mod->Lookup(gvar));
    auto new_func =
        Function(func->params, VisitExpr(func->body), func->ret_struct_info, func->attrs);
    builder_->UpdateFunction(gvar, new_func);
    return RemoveUnusedFunctions(builder_->GetContextIRModule(), {"main"});
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    if (auto const* gvar = call_node->op.as<GlobalVarNode>()) {
      auto func = builder_->GetContextIRModule()->Lookup(GetRef<GlobalVar>(gvar));
      if (auto composite_name = func->GetAttr<String>(attr::kComposite)) {
        auto new_func = Downcast<Function>(VisitExpr(func));
        auto codegen_name = GetCodegenName(composite_name.value());
        auto gsymbol = gvar->name_hint + "_" + codegen_name;
        new_func = WithAttr(new_func, attr::kCodegen, codegen_name);
        new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol, gsymbol);
        auto new_gvar = builder_->AddFunction(new_func, gsymbol);
        return Call(new_gvar, call_node->args);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto f_inner = ExprMutator::VisitExpr_(func_node);
    auto composite_name = func_node->GetAttr<String>(attr::kComposite);
    ICHECK(composite_name);

    Array<Var> param_vars;
    Array<Expr> params;

    for (auto v : func_node->params) {
      Var new_v(v->name_hint(), GetStructInfo(v));
      param_vars.push_back(new_v);
      params.push_back(new_v);
    }

    return Function(param_vars, Call(f_inner, params), func_node->ret_struct_info);
  }

 private:
  String GetCodegenName(const std::string& composite_name) {
    auto delim_pos = composite_name.find(".");
    ICHECK(delim_pos != std::string::npos) << "The pattern name for a composite function should "
                                              "start with a compiler name followed by period.";
    return composite_name.substr(0, delim_pos);
  }
};

namespace transform {

Pass WrapCompositeFunction() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return WrapComposite(mod).Run(); };
  return CreateModulePass(pass_func, 0, "WrapCompositeFunction", {});
}

TVM_REGISTER_GLOBAL("relax.transform.WrapCompositeFunction").set_body_typed(WrapCompositeFunction);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
