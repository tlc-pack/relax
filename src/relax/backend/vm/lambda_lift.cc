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
 * \file tvm/relax/backend/vm/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/node/structural_equal.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {
namespace relax_vm {

using namespace tvm::runtime;

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const IRModule& module) : module_(module) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto global_var_node = call_node->op.as<GlobalVarNode>()) {
      String rec_name = global_var_node->name_hint;
      auto global_var = GetRef<GlobalVar>(global_var_node);
      auto it = lambda_map_.find(global_var);
      ICHECK(it != lambda_map_.end());
      return Call(it->second, call->args, call_node->attrs, call_node->type_args);
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // We should not transform primitive functions.
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return std::move(func);
    }

    auto name = std::string("lifted_func_") + std::to_string(lift_func_num_++);
    auto global = GlobalVar(name);
    auto free_vars = FreeVars(func);
    auto rec_vars = RecGlobalVars(func);
    auto all_global_vars = AllGlobalVars(func);

    for (const auto& var : rec_vars) {
      recur_vars_.push_back(var);
    }

    Array<Var> captured_vars;
    bool recursive = false;
    for (const auto& var : free_vars) {
      captured_vars.push_back(var);
    }
    if (!rec_vars.empty()) {
      recursive = true;
    }
    Array<Var> typed_captured_vars;
    Map<Var, Expr> rebinding_map;
    for (auto free_var : captured_vars) {
      auto var = Var(free_var->name_hint(), NullOpt, free_var->type_annotation);
      typed_captured_vars.push_back(var);
      rebinding_map.Set(free_var, var);
    }
    if (recursive) {
      if (!captured_vars.empty()) {
        Array<Expr> fvs;
        for (auto fv : captured_vars) {
          fvs.push_back(fv);
        }
        lambda_map_.emplace(recur_vars_.back(), Call(global, fvs));
      } else {
        if (recur_vars_.size() > 0) {
          lambda_map_.emplace(recur_vars_.back(), global);
        }
      }
    }

    auto body = Downcast<Function>(ExprMutator::VisitExpr_(func_node));

    Function lifted_func;
    if (captured_vars.size() == 0) {
      lifted_func = Function(global, body->params, body->body, body->ret_type, body->span);
    } else {
      auto before = Downcast<Function>(body)->params.size();
      auto inner_name = std::string("inner_func_") + std::to_string(inner_func_num_++);
      auto rebound_body =
          Function(GlobalVar(inner_name), func->params, body->body, func->ret_type, func->span);
      auto after = Downcast<Function>(rebound_body)->params.size();
      CHECK_EQ(before, after);
      lifted_func = Function(global, captured_vars, rebound_body, func->checked_type_);
      // todo (@yongwww): call make_closure intrinsic here
    }

    ICHECK(lifted_func.defined());

    if (module_->ContainGlobalVar(name)) {
      const auto existing_func = module_->Lookup(name);
      ICHECK(tvm::StructuralEqual()(lifted_func, existing_func))
          << "lifted function hash collision";
      // If an identical function already exists, use its global var.
      global = module_->GetGlobalVar(name);
    } else {
      // Add the lifted function to the module.
      module_->Add(global, lifted_func);
    }

    if (captured_vars.size() == 0) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
        fvs.push_back(fv);
      }
      return Call(global, fvs);
    }
  }

  IRModule Lift() {
    auto glob_funcs = module_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->name, func->params, VisitExpr(func->body), func->ret_type);
        module_->Add(pair.first, func, true);
      }
    }
    return module_;
  }

 private:
  std::unordered_map<GlobalVar, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  std::vector<GlobalVar> recur_vars_;
  IRModule module_;
  size_t lift_func_num_ = 0;
  size_t inner_func_num_ = 0;
};

}  // namespace relax_vm

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::relax_vm::LambdaLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LambdaLift").set_body_typed(LambdaLift);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
