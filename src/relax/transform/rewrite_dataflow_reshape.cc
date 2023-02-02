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
 * \file src/relax/transform/rewrite_dataflow_reshape.cc
 * \brief Transform all reshape within dataflow block to a specialized reshape operator
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class DataflowReshapeRewriter : public ExprMutator {
 public:
  explicit DataflowReshapeRewriter(const IRModule& mod) : mod_(mod) {}

 private:
  BindingBlock VisitBindingBlock(const BindingBlock& block) final {
    // We only rewrite the bindings inside dataflow blocks.
    if (const auto* dataflow_block = block.as<DataflowBlockNode>()) {
      return VisitBindingBlock_(dataflow_block);
    } else {
      return block;
    }
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    // We only rewrite the bindings that are not dataflow output (which means they are not
    // externally referenced)
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      this->builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      ExprMutator::VisitBinding_(binding);
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    if (!IsCallingTIRReshape(call)) {
      return GetRef<Call>(call);
    }
    static const ExternFunc& vm_builtin_reshape = ExternFunc("vm.builtin.reshape");
    Array<Expr> args = Downcast<Tuple>(call->args[1])->fields;
    ICHECK_EQ(args.size(), 1);
    TensorStructInfo res_sinfo = Downcast<TensorStructInfo>(call->struct_info_);
    ShapeExpr new_shape = Downcast<ShapeExpr>(res_sinfo->shape);
    return Call(vm_builtin_reshape, {args[0], new_shape}, Attrs(), {res_sinfo});
  }

  bool IsCallingTIRReshape(const CallNode* call) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op != call_tir_op) {
      return false;
    }
    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    const auto* func = mod_->functions.Get(gv).as<tir::PrimFuncNode>();
    ICHECK_NOTNULL(func);
    return HasReshapePattern(GetRef<tir::PrimFunc>(func));
  }

  const IRModule& mod_;
};

Expr RewriteDataflowReshape(const Function& f, const IRModule& mod) {
  return DataflowReshapeRewriter(mod)(f);
}

namespace transform {

Pass RewriteDataflowReshape() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(RewriteDataflowReshape(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "RewriteDataflowReshape", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteDataflowReshape")
    .set_body_typed(RewriteDataflowReshape);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
