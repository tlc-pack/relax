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
 * \file src/relax/transform/autodiff.cc
 * \brief Auto-differentiation
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relax {

// ==================
// ReverseModeAD:
// Reverse Mode Auto-differentiation.

class ReverseModeADMutator : public ExprMutator {
  public:
    explicit ReverseModeADMutator(IRModule mod) { mod_ = mod; }

    IRModule Lower() {
      for (auto& p : mod_->functions) {
        Expr func = p.second;
        if (func->IsInstance<FunctionNode>()) {
          func = this->VisitExpr(func);
        }
        builder_->AddFuncToContext(Downcast<BaseFunc>(func), p.first->name_hint);
      }
      return builder_->GetContextIRModule();
    }

    void Emit(const Var& v, const Expr& e) {
      e->checked_type_ = v->checked_type();
      e->shape_ = v->shape();
      VarBinding node = VarBinding(v, e);
      if (node->var->IsInstance<DataflowVarNode>()) {
        this->builder_->Emit(node);
      } else {
        this->builder_->EmitOutput(node);
      }
    }

    Expr VisitExpr_(const FunctionNode* node) override {
      adjoint_map.clear();
      gradient_map.clear();

      ICHECK(node->body->IsInstance<SeqExprNode>());
      const SeqExprNode* body = node->body.as<SeqExprNode>();

      ICHECK(body->blocks.size() == 1);
      ICHECK(body->blocks[0]->IsInstance<DataflowBlockNode>());

      // TODO: body can be single var or tuple of vars, support tuple
      ICHECK(body->body->IsInstance<VarNode>());

      for (const auto& v: node->params) {
        Var adjoint_v = Var(v->name_hint() + "`", v->shape(), v->checked_type());
				adjoint_v->checked_type_ = v->checked_type();
        adjoint_map.Set(v, adjoint_v);
      }

      builder_->BeginDataflowBlock();
      for (auto binding : body->blocks[0]->bindings) {
        this->VisitBinding(binding);
      }

      for (int i = body->blocks[0]->bindings.size() - 1; i >= 0; i--) {
        const Binding& binding = body->blocks[0]->bindings[i];
        if (!binding->IsInstance<VarBindingNode>()) {
          continue;
        }

        const VarBindingNode* var_binding = binding.as<VarBindingNode>();
        
        if (gradient_map.count(var_binding->var) == 0) {
          // var_binding->var is the derivative with respect to
          // so it should be an output
          ICHECK(!var_binding->var->IsInstance<DataflowVarNode>());

          Op op = var_binding->var == body->body ? Op::Get("ones_like") : Op::Get("zeros_like");

          this->Emit(adjoint_map[var_binding->var], Call(op, {var_binding->var}));
        } else {
          Array<Expr> partials = gradient_map[var_binding->var];
          ICHECK(partials.size() != 0);

          const Op add_op = Op::Get("add");

          // sum the partials
          Expr sum = partials[0];
          for (size_t i = 1; i < partials.size(); i++) {
            sum = Call(add_op, {sum, partials[i]});
          }

          this->Emit(adjoint_map[var_binding->var], sum);
        }
      }
      for (const auto& v: node->params) {
          Array<Expr> partials = gradient_map[v];
          ICHECK(partials.size() != 0);

          const Op add_op = Op::Get("add");

          // sum the partials
          Expr sum = partials[0];
          for (size_t i = 1; i < partials.size(); i++) {
            sum = Call(add_op, {sum, partials[i]});
          }

          this->Emit(adjoint_map[v], sum);
			}

      Array<Expr> body_out;
      Array<Expr> shape;
      Array<Type> func_ret_type;
      body_out.push_back(body->body);
      shape.push_back(body->body->shape());
      func_ret_type.push_back(node->ret_type);
      for (const auto& v: node->params) {
        const auto& adjoint_v = adjoint_map[v];
        body_out.push_back(adjoint_v);
        shape.push_back(v->shape());
        func_ret_type.push_back(adjoint_v->checked_type());
      }

      Var out = Var("out", Tuple(shape), TupleType(func_ret_type));
      
      this->Emit(out, Tuple(body_out));

      BindingBlock block = builder_->EndBlock();
      return Function(node->name, node->params, SeqExpr({block}, out), TupleType(func_ret_type));
    }

    void VisitBinding_(const VarBindingNode* binding) override {
      ExprMutator::VisitBinding_(binding);

      Var v = binding->var;
      if (adjoint_map.count(v) == 0) {
        Var adjoint_v = DataflowVar(v->name_hint() + "`", v->shape(), v->checked_type());
				adjoint_v->checked_type_ = v->checked_type();
				adjoint_map.Set(v, adjoint_v);
      }

      if (binding->value->IsInstance<VarNode>()) {
        const VarNode* node = binding->value.as<VarNode>();
        if (gradient_map.count(GetRef<Var>(node)) == 0) {
          gradient_map.Set(GetRef<Var>(node), Array<Expr>());
        }

        Array<Expr> partials = gradient_map[GetRef<Var>(node)];
        partials.insert(partials.end(), adjoint_map[v]);
        gradient_map.Set(GetRef<Var>(node), partials);
        return;
      }
      ICHECK(binding->value->IsInstance<CallNode>());
      const CallNode* node = binding->value.as<CallNode>();
      const OpNode* op = node->op.as<OpNode>();
      ICHECK(op != nullptr);

      Op op_ref = GetRef<Op>(op);
      ICHECK(rev_map.count(op_ref) != 0);
      
      Array<Expr> rev = rev_map[op_ref](GetRef<Call>(node), adjoint_map[v]);
      ICHECK(rev.size() == node->args.size());

      for (size_t i = 0; i < node->args.size(); i++) {
        const VarNode* arg = node->args[i].as<VarNode>();
        ICHECK(arg != nullptr);

        if (gradient_map.count(GetRef<Var>(arg)) == 0) {
          gradient_map.Set(GetRef<Var>(arg), Array<Expr>());
        }

        Array<Expr> partials = gradient_map[GetRef<Var>(arg)];
        partials.insert(partials.end(), rev[i]);
        gradient_map.Set(GetRef<Var>(arg), partials);
      }
    }

  private:
    IRModule mod_;
    // map var -> adjoint
    Map<Var, Var> adjoint_map;
    // map var -> array of partial gradients that sum up to equal the adjoint
    Map<Var, Array<Expr>> gradient_map;

    const OpAttrMap<relay::FPrimalGradient> rev_map = Op::GetAttrMap<relay::FPrimalGradient>("FPrimalGradient");
};

namespace transform {

Pass ReverseModeAD() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return ReverseModeADMutator(mod).Lower(); };
  return CreateModulePass(pass_func, 0, "ReverseModeAD", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ReverseModeAD").set_body_typed(ReverseModeAD);

}  // namespace transform

}
}
