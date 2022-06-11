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
 * \file src/relax/transform/resolve_globals.cc
 * \brief Resolve GlobalVars using string equality.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class GlobalVarResolver : public ExprMutator {
 public:
  GlobalVarResolver(IRModule mod, DiagnosticContext diag_ctx) : mod_(mod), diag_ctx_(diag_ctx) {}

  Expr VisitExpr_(const GlobalVarNode* gvar) {
    if (!mod_->ContainGlobalVar(gvar->name_hint)) {
      return GetRef<GlobalVar>(gvar);
    }
    return mod_->GetGlobalVar(gvar->name_hint);
  }

 private:
  /*! \brief the IRModule used for GlobalVar lookup. */
  IRModule mod_;
  DiagnosticContext diag_ctx_;
};

namespace transform {

Pass ResolveGlobals() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [](Function f, IRModule m, PassContext pc) {
        // TODO(@altanh): make sure pc always has diag_ctx?
        GlobalVarResolver resolver(m, pc->diag_ctx.value());
        return Downcast<Function>(resolver.VisitExpr(f));
      };
  return CreateFunctionPass(pass_func, 0, "ResolveGlobals", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ResolveGlobals").set_body_typed(ResolveGlobals);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
