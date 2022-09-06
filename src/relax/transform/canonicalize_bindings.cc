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
 * \file tvm/relax/transform/canonicalize.cc
 * \brief Pass for simplifying modules by folding var bindings and match shape nodes.
 *        May include other forms of simplification in the future.
 *        Ideally should be used before constant folding and eliminating unused bindings.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class BindingCanonicalizer : public ExprMutator {
 public:
  BindingCanonicalizer() {}

  Expr VisitExpr_(const VarNode* op) override {
    // remap first
    Var v = Downcast<Var>(ExprMutator::VisitExpr_(op));
    if (!CanCanonicalizeVar(v)) {
      return Downcast<Expr>(v);
    }
    // visit again in case we need to do a substitution in the value
    return ExprMutator::VisitExpr_(LookupBinding(v).as<VarNode>());
  }

  Expr VisitExpr_(const DataflowVarNode* op) override {
    Var v = Downcast<Var>(ExprMutator::VisitExpr_(op));
    if (!CanCanonicalizeVar(v)) {
      return Downcast<Expr>(v);
    }
    return ExprMutator::VisitExpr_(LookupBinding(v).as<DataflowVarNode>());
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    // Unlike default visitor, preserve the checked_type_
    // We may need to change the shape field in case there are substitutions
    // that need to be performed within the shape computation.
    Expr new_value = this->VisitExpr(binding->value);
    Var new_var = this->VisitVarDef(binding->var);

    auto emit = [this](VarBinding b) {
      if (this->builder_->CurrentBlockIsDataFlow() && !b->var.as<DataflowVarNode>()) {
        this->builder_->EmitOutput(b);
      } else {
        this->builder_->Emit(b);
      }
    };

    if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
      emit(GetRef<VarBinding>(binding));
      return;
    }

    // we don't look at the new value's shape or checked type; we only consider
    // if there were any substitutions performed within the original var's shape_
    Var temp = WithShapeAndType(new_var, new_var->shape_, new_var->checked_type_);
    if (!temp.same_as(new_var)) {
      new_var = temp;
      this->var_remap_[binding->var->vid] = new_var;
    }

    // unlike default visitor, we do not permit the var's checked_type to change
    emit(VarBinding(new_var, new_value));
  }

  void VisitBinding_(const MatchShapeNode* binding) override {
    // For match shape, we need to be cleverer and allow the shape_ to change
    // due to possible substitutions.
    // Additionally, if we have a trivial shape check (the shape_ of LHS and RHS is the same),
    // we can canonicalize to a var binding
    Expr new_value = this->VisitExpr(binding->value);
    Expr new_pattern = this->VisitExpr(ShapeExpr(binding->pattern));

    Var new_var;
    if (binding->var.defined()) {
      Optional<Expr> new_shape;
      if (new_value->checked_type_.as<DynTensorTypeNode>()) {
        new_shape = new_pattern;
      }
      // visit var def visits the var's shape_ field and may perform variable substitutions,
      // so we should use that shape_ if it's defined
      new_var = this->VisitVarDef(binding->var);
      if (new_var->shape_.defined()) {
        new_shape = Downcast<Expr>(new_var->shape_);
      }

      // do not permit the type to change
      Var temp = WithShapeAndType(new_var, new_shape, binding->var->checked_type_);
      if (!temp.same_as(new_var)) {
        new_var = temp;
        this->var_remap_[binding->var->vid] = new_var;
      }
    }

    // if the LHS and RHS have the same shape_, we canonicalize to a var binding instead
    if (new_var.defined() && new_value->shape_.defined() &&
        builder_->CanProveShapeEqual(Downcast<Expr>(new_var->shape_),
                                     Downcast<Expr>(new_value->shape_))) {
      builder_->Emit(VarBinding(new_var, new_value));
      return;
    }

    // reemit old binding if nothing changes
    if (new_value.same_as(binding->value) && new_pattern.same_as(binding->pattern)) {
      if (!binding->var.defined() || (binding->var.defined() && new_var.same_as(binding->var))) {
        builder_->EmitMatchShape(GetRef<MatchShape>(binding));
        return;
      }
    }

    builder_->EmitMatchShape(
        MatchShape(new_value, Downcast<ShapeExpr>(new_pattern)->values, new_var));
  }

 private:
  bool AnnotationsDiffer(const ObjectRef& obj1, const ObjectRef& obj2,
                         std::function<bool(const ObjectRef&, const ObjectRef&)> check_eq) {
    // annotations differ if one is present but not the other
    // or they're both present and they differ
    bool both_present = obj1.defined() && obj2.defined();
    bool neither_present = !obj1.defined() && !obj2.defined();
    return !(both_present || neither_present) || (both_present && !check_eq(obj1, obj2));
  }

  bool CanCanonicalizeVar(Var v) {
    Optional<Expr> value = LookupBinding(v);
    // can replace only if the value is also a var
    if (!value || !value.as<VarNode>()) {
      return false;
    }
    Var parent_var = Downcast<Var>(value);

    // Cases when we conservatively do not unify:
    // 1. checked_type_ or shape_ of the child differs from that of the parent
    //    In this case, we could be overriding user annotations.
    // 2. If the child is a Var and the parent is a DataflowVar.
    //    That could result in a DataflowVar leaving the current DataflowBlock.
    bool annotations_differ =
        AnnotationsDiffer(v->shape_, parent_var->shape_,
                          [&](const ObjectRef& shape1, const ObjectRef& shape2) {
                            return builder_->CanProveShapeEqual(Downcast<Expr>(shape1),
                                                                Downcast<Expr>(shape2));
                          }) ||
        AnnotationsDiffer(v->checked_type_, parent_var->checked_type_,
                          [&](const ObjectRef& type1, const ObjectRef& type2) {
                            return tvm::StructuralEqual()(type1, type2);
                          });
    bool var_to_dataflow = (!v.as<DataflowVarNode>() && parent_var.as<DataflowVarNode>());
    return !annotations_differ && !var_to_dataflow;
  }
};

Expr CanonicalizeBindings(const Expr& e) { return BindingCanonicalizer().VisitExpr(e); }

namespace transform {

Pass CanonicalizeBindings() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CanonicalizeBindings(f));
      };
  return CreateFunctionPass(pass_func, 1, "CanonicalizeBindings", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CanonicalizeBindings").set_body_typed(CanonicalizeBindings);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
