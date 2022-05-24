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
 * \file tvm/relax/transform/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const IRModule& module) : module_(std::move(module)) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto const* var = call_node->op.as<relax::VarNode>()) {
      bool has_closure = HasClosure(GetRef<Var>(var));
      auto val = builder_->LookupBinding(GetRef<Var>(var));
      // Call "relax.invoke_closure" to invoke closure
      if (has_closure && val.as<CallNode>()) {
        Var clo_arg = GetRef<Var>(var);
        if (this->var_remap_.find(var->vid) != this->var_remap_.end()) {
          clo_arg = this->var_remap_.at(var->vid);
        }
        return Call(invoke_closure_op_, {clo_arg, Tuple(call_node->args)}, {},
                    {call_node->checked_type_});
      }
    }
    if (auto global_var_node = call_node->op.as<GlobalVarNode>()) {
      String rec_name = global_var_node->name_hint;
      auto global_var = GetRef<GlobalVar>(global_var_node);
      auto it = lambda_map_.find(global_var);
      if (it != lambda_map_.end()) {
        // flatten nested call, e.g. call(y)(x) -> call(x, y))
        Array<relay::Expr> new_args;
        for (const auto arg : call->args) {
          new_args.push_back(arg);
        }
        if (const auto* nest_call = it->second.as<CallNode>()) {
          for (const auto arg : nest_call->args) {
            new_args.push_back(arg);
          }
          return Call(nest_call->op, new_args, call_node->attrs, call_node->type_args);
        }
        return Call(it->second, call->args, call_node->attrs, call_node->type_args);
      }
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // We should not transform primitive functions.
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return std::move(func);
    }

    String lift_func_name = std::string("lifted_func_") + std::to_string(lift_func_num_++);
    auto global = GlobalVar(lift_func_name);
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
      Var var = Var(free_var->name_hint(), NullOpt, free_var->checked_type_, free_var->span);
      var->shape_ = free_var->shape_;
      typed_captured_vars.push_back(var);
      rebinding_map.Set(free_var, var);
    }

    // recursive call
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
    Map<String, ObjectRef> attrs;
    attrs.Set(tvm::attr::kGlobalSymbol, lift_func_name);

    if (captured_vars.size() == 0) {
      lifted_func = Function(
          /*params=*/body->params,
          /*body=*/body->body,
          /*ret_type=*/body->ret_type,
          /*attrs=*/DictAttrs(attrs));
    } else {
      // Inner function before Flattening
      // String inner_name = "inner_func_" + inner_func_num_++;
      // attrs.Set(tvm::attr::kGlobalSymbol, inner_name);
      // auto before = Downcast<Function>(body)->params.size();
      // auto rebound_body = Function(/*params=*/func->params,
      //                              /*body=*/Bind(body->body, rebinding_map),
      //                              /*ret_type=*/func->ret_type,
      //                              /*attrs=*/DictAttrs(attrs),
      //                              /*span=*/func->span);
      // auto after = Downcast<Function>(rebound_body)->params.size();
      // CHECK_EQ(before, after);
      // attrs.Set(tvm::attr::kGlobalSymbol, lift_func_name);
      // lifted_func = Function(/*params=*/typed_captured_vars,
      //                        /*body=*/rebound_body,
      //                        /*ret_type=*/func->checked_type_,
      //                        /*attrs=*/DictAttrs(attrs),
      //                        /*span=*/func->span);

      // Flatten the Closure
      std::vector<Var> closure_params;
      closure_params.reserve(func->params.size() + typed_captured_vars.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        closure_params.emplace_back(func->params[i]);
      }
      for (size_t i = 0; i < typed_captured_vars.size(); ++i) {
        closure_params.emplace_back(typed_captured_vars[i]);
      }

      lifted_func = Function(/*params=*/closure_params,
                             /*body=*/Bind(body->body, rebinding_map),
                             /*ret_type=*/func->ret_type,
                             /*attrs=*/DictAttrs(attrs),
                             /*span=*/func->span);

      Array<Type> param_types;
      for (Var param : closure_params) {
        CHECK(param->checked_type_.defined())
            << "relax.Function requires params to contain checked_type_";
        param_types.push_back(param->checked_type_);
      }
      lifted_func->checked_type_ = FuncType(param_types, body->ret_type, {}, {});
    }

    ICHECK(lifted_func.defined());

    auto ctx_mod = builder_->GetContextIRModule();

    if (ctx_mod->ContainGlobalVar(lift_func_name)) {
      const auto existing_func = ctx_mod->Lookup(lift_func_name);
      ICHECK(tvm::StructuralEqual()(lifted_func, existing_func))
          << "lifted function hash collision";
      // If an identical function already exists, use its global var.
      global = ctx_mod->GetGlobalVar(lift_func_name);
    } else {
      // Add the lifted function to the module.
      builder_->UpdateFunction(global, lifted_func);
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
      // Call make_closure intrinsic
      return Call(make_closure_op_, {global, Tuple(fvs)}, {}, {});
    }
  }

  bool HasClosure(const Var& var) {
    auto val = builder_->LookupBinding(var);
    if (const auto* value = val.as<GlobalVarNode>()) {
      IRModule ctx_mod = builder_->GetContextIRModule();
      ICHECK(ctx_mod->functions.size() > 0);
      BaseFunc func = ctx_mod->Lookup(GetRef<GlobalVar>(value));
      if (const auto* func_node = func.as<FunctionNode>()) {
        if (const auto* call_node = func_node->body.as<CallNode>()) {
          if (call_node->op == make_closure_op_) {
            return true;
          }
        }
      }
    } else if (const auto* func_node = val.as<FunctionNode>()) {
      if (const auto* call_node = func_node->body.as<CallNode>()) {
        if (call_node->op == make_closure_op_) {
          return true;
        }
      }
    } else if (const auto* call_node = val.as<relax::CallNode>()) {
      // recursive call
      auto op = call_node->op;
      if (make_closure_op_ == op) {
        return true;
      }
      if (const auto* lv = op.as<VarNode>()) {
        return HasClosure(GetRef<Var>(lv));
      }
    }
    return false;
  }

  IRModule Lift() {
    auto glob_funcs = module_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->attrs);
        builder_->UpdateFunction(pair.first, func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  std::unordered_map<GlobalVar, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  std::vector<GlobalVar> recur_vars_;
  IRModule module_;
  size_t lift_func_num_ = 0;
  /*! \brief Cache ops that would be used later to reduce lookup overhead. */
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
};

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::LambdaLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LambdaLift").set_body_typed(LambdaLift);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
