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
 * \file tvm/relax/transform/to_anf.cc
 * \brief Pass for transforming Relax IR to A-normal form.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relax {


// TODO(@altanh): LCA binding lifting
class ToANFMutator : public ExprMutator {
 public:
  ToANFMutator(const IRModule& mod) : mod_(mod) {}

  IRModule Lower() {
    IRModule ret_mod = IRModule();
    for (auto& p : mod_->functions) {
      Expr func = p.second;
      if (p.second->IsInstance<FunctionNode>()) {
        func = this->VisitExpr(p.second);
      }
      ret_mod->Add(p.first, Downcast<BaseFunc>(func));
    }
    return ret_mod;
  }

 private:
  IRModule mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.to_anf").set_body_typed([](IRModule mod) {
  return ToANFMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
