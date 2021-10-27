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

// TODO(@altanh): CopyOnWrite

class ToANFMutator : public ExprMutator {
 public:
  ToANFMutator(const IRModule& mod) : mod_(mod) {}

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

  Expr VisitExpr(const Expr& n) final {
    return builder_->Normalize(ExprMutator::VisitExpr(n));
  }

  // Expr VisitExpr_(const TupleNode* op) override {
  //   Array<Expr> new_fields;
  //   for (const Expr& field : op->fields) {
  //     new_fields.push_back(Bind(field));
  //   }
  //   return Tuple(new_fields);
  // }

  // Expr VisitExpr_(const CallNode* op) override {
  //   Expr new_op = Bind(op->op);
  //   Array<Expr> new_args;
  //   for (const Expr& arg : op->args) {
  //     new_args.push_back(Bind(arg));
  //   }
  //   return Call(new_op, new_args, op->attrs, op->type_args);
  // }

  // Expr VisitExpr_(const TupleGetItemNode* op) {
  //   Expr new_tuple = Bind(op->tuple);
  //   return TupleGetItem(new_tuple, op->index);
  // }

 private:
  // Expr Bind(const Expr& expr) {
  //   Expr post = this->Mutate(expr);
  //   if (IsLeaf(post)) {
  //     return post;
  //   }
  //   ICHECK(!expr.as<VarNode>());
  //   Expr var = builder_->Emit(post);
  //   expr_memo_[expr] = var;
  //   return var;
  // }

  // bool IsLeaf(const Expr& expr) {
  //   return expr.as<VarNode>() || expr.as<GlobalVarNode>() || expr.as<relay::ConstantNode>() ||
  //          expr.as<ShapeExprNode>() || expr.as<ExternFuncNode>() || expr.as<OpNode>();
  // }

  IRModule mod_;
};

TVM_REGISTER_GLOBAL("relax.transform.to_anf").set_body_typed([](IRModule mod) {
  return ToANFMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
