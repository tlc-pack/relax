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
 * \file src/relax/transform/to_non_dataflow.cc
 * \brief
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

class ToNonDFMutator : public ExprMutator {
 public:
  explicit ToNonDFMutator(IRModule mod) { mod_ = mod; }

  IRModule Lower() {
    IRModule ret_mod = IRModule();
    for (auto& p : mod_->functions) {
      Expr func = p.second;
      if (p.second->IsInstance<FunctionNode>()) {
        func = this->Mutate(p.second);
      }
      ret_mod->Add(p.first, Downcast<BaseFunc>(func));
    }
    return ret_mod;
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    return Var(op->vid, op->shape(), op->type_annotation, op->span);
  }

  BindingBlock VisitDataflowBlock(const DataflowBlock& block) final {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

 private:
  IRModule mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.to_non_dataflow").set_body_typed([](IRModule mod) {
  return ToNonDFMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
