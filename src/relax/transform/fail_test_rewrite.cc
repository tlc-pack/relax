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
 * \file src/relax/transform/fail_test_rewrite.cc
 * \brief Incorrectly transform the dataflow structure as fail testcases.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

/*! \brief Rewrite/Remove global var or symbolic var in the dataflow block.*/
class FailTestRewriter : public ExprMutator {
  using ExprMutator::VisitExpr_;

  // Rewrite/Remove specific global var
  Var VisitVarDef_(const VarNode* var) override {
    if (var->name_hint() == "gv_rewrite") {
      Var new_var = Var("gv_rewrite", /*struct_info_annotation=*/NullOpt);
      return std::move(new_var);
    } else if (var->name_hint() == "gv_remove") {
      Var new_var = Var("new_gv", /*struct_info_annotation=*/NullOpt);
      return std::move(new_var);
    }
    return GetRef<Var>(var);
  }

  // Rewrite/Remove specific symbolic var
  Expr VisitExpr_(const ShapeExprNode* op) override {
    if (op->values.size() == 2) {
      tir::Var arg0 = Downcast<tir::Var>(op->values[0]);
      tir::Var new_arg0 = tir::Var(arg0->name_hint);
      ShapeExpr new_op = ShapeExpr({new_arg0, op->values[1]});
      return std::move(new_op);
    } else if (op->values.size() == 3) {
      ShapeExpr new_op = ShapeExpr({op->values[0], op->values[1]});
      return std::move(new_op);
    }
    return GetRef<Expr>(op);
  }
};

BindingBlock FailTestRewrite(const BindingBlock& block) {
  return FailTestRewriter().VisitBindingBlock(block);
}

namespace transform {

Pass FailTestRewrite() {
  runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func =
      [=](DataflowBlock block, IRModule m, PassContext pc) {
        return Downcast<DataflowBlock>(FailTestRewrite(block));
      };
  return CreateDataflowBlockPass(pass_func, 2, "FailTestRewrite", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FailTestRewrite").set_body_typed(FailTestRewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
