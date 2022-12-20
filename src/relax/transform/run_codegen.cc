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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/run_codegen.cc
 * \brief Run codegen for annotated relax functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

#include <iostream>

namespace tvm {
namespace relax {

class CodeGenRunner : ExprMutator {
 public:
  explicit CodeGenRunner(IRModule mod, Optional<Array<runtime::String>> target_codegens,
                         Array<runtime::String> entry_functions)
      : ExprMutator(mod), entry_functions_(std::move(entry_functions)) {
    if (target_codegens.defined()) {
      for (auto target : target_codegens.value()) {
        target_codegens_.insert(target);
      }
    }
  }

  IRModule Run() {
    IRModule mod = builder_->GetContextIRModule();
    for (const String& entry_func_name : entry_functions_) {
      auto entry_func = mod->Lookup(entry_func_name);
      auto gvar = mod->GetGlobalVar(entry_func_name);
      builder_->UpdateFunction(gvar, Downcast<BaseFunc>(VisitExpr(entry_func)));
    }

    IRModule out_mod = builder_->GetContextIRModule();
    if (ext_mods_.size()) {
      out_mod = WithAttr(out_mod, "external_mods", std::move(ext_mods_));
    }

    return out_mod;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) override {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto const* gvarnode = call_node->op.as<GlobalVarNode>()) {
      const GlobalVar gvar = GetRef<GlobalVar>(gvarnode);
      // TODO(@sunggg): Is there any better way to get this func?
      Function func = Downcast<Function>(builder_->GetContextIRModule()->Lookup(gvar));
      Expr new_op = VisitExpr(func);
      if (new_op->IsInstance<ExternFuncNode>()) {
        Array<Expr> new_args({new_op});
        Array<Expr> tmp_args;
        for (const auto& arg : call_node->args) {
          tmp_args.push_back(VisitExpr(arg));
        }
        new_args.push_back(Tuple(tmp_args));
        new_args.push_back(func->body->shape());

        static const Op& call_op = Op::Get("relax.call_tir");

        // Remove global symbol and codegen from the function so that it can be removed.
        static const runtime::PackedFunc* RemoveFuncAttrFunc =
            runtime::Registry::Get("ir.BaseFuncWithoutAttr");
        ICHECK(RemoveFuncAttrFunc);
        func = (*RemoveFuncAttrFunc)(func, tvm::attr::kGlobalSymbol);
        func = (*RemoveFuncAttrFunc)(func, attr::kCodegen);
        builder_->UpdateFunction(gvar, func);

        return Call(call_op, new_args, tvm::Attrs(), {GetStaticType(func->ret_struct_info)});
      }
    }
    Array<Expr> new_args;
    for (const auto& arg : call_node->args) {
      new_args.push_back(VisitExpr(arg));
    }

    return Call(call_node->op, new_args, call_node->attrs, call_node->type_args, call_node->span);
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    Function func = GetRef<Function>(func_node);
    auto opt_codegen = func->GetAttr<String>(attr::kCodegen);
    if (opt_codegen.defined()) {
      auto opt_gsymbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(opt_gsymbol.defined())
          << "When a codegen is defined, global symbol should be defined together.";

      String codegen_str = opt_codegen.value();
      // If the current codegen is not in the provided target lists, defer the codegen process.
      if (target_codegens_.size() && target_codegens_.count(codegen_str) == 0) {
        return GetRef<Function>(func_node);
      }

      // Start the codegen process.
      // Get the codegen with its ffi key.
      String codegen_name = "relax.ext." + codegen_str;
      auto codegen = runtime::Registry::Get(codegen_name);
      ICHECK(codegen) << "Codegen is not found: " << codegen_name << "\n";
      // Store the produced output runtime module in the internal array.
      ext_mods_.push_back((*codegen)(func));

      // Return the external function with given global symbol.
      return ExternFunc(opt_gsymbol.value());
    } else {
      return ExprMutator::VisitExpr_(func_node);
    }
  }

 private:
  Array<runtime::String> entry_functions_;
  std::unordered_set<std::string> target_codegens_;
  Array<runtime::Module> ext_mods_;
};

}  // namespace relax

namespace transform {
Pass RunCodegen(Optional<Array<runtime::String>> target_codegens,
                Array<runtime::String> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    return relax::CodeGenRunner(m, target_codegens, entry_functions).Run();
  };
  return CreateModulePass(pass_func, 0, "RunCodegen", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RunCodegen").set_body_typed(RunCodegen);

}  // namespace transform
}  // namespace tvm
