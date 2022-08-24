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
 * \brief Pass for simplifying modules by coalescing intermediate bindings.
 *        May include other forms of simplification in the future.
 *        Ideally should be used before constant folding and eliminating unused bindings.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

// Simple union-find implementation with path compression.
// Anchors are variables not assigned to other variables;
// we track them separately to ensure an ordering to the matches.
// E.g., if we have
//   x = constant
//   y = x
//   z = y
// We might get unlucky in some situations and have z end up being the parent,
// then if we naively do the replacement, we get
//   x = constant
//   y = z << bug
//   z = z
class VarUnifier {
 public:
  explicit VarUnifier() : parent_map_(), anchors_() {}

  void InsertAnchor(const Var& v) {
    CHECK(!Contains(v));
    anchors_.emplace(v);
    parent_map_.Set(v, v);
  }

  void InsertBinding(const Var& v, const Var& anchor) {
    CHECK(Contains(anchor));
    CHECK(!Contains(v));
    Var parent = Lookup(anchor);
    CHECK(anchors_.count(parent));
    parent_map_.Set(v, parent);
  }

  bool Contains(const Var& v) { return parent_map_.count(v) != 0; }

  Var Lookup(const Var& v) {
    CHECK(Contains(v));
    Var parent = parent_map_.at(v);
    if (v.same_as(parent)) {
      return v;
    }
    Var ret = Lookup(parent);
    // path compression
    parent_map_.Set(v, ret);
    return ret;
  }

  void Unify(const Var& v1, const Var& v2) {
    Var p1 = Lookup(v1);
    Var p2 = Lookup(v2);
    if (p1.same_as(p2)) {
      return;
    }
    // we could be clever and check the depth of the trees if performance is an issue
    parent_map_.Set(p1, p2);
  }

 private:
  Map<Var, Var> parent_map_;
  // anchors: vars that are bound to a non-var. A parent var must be an anchor
  std::set<Var> anchors_;
};

class Canonicalizer : public ExprMutator {
 public:
  explicit Canonicalizer() : uf_() {}

  Expr VisitExpr_(const VarNode* op) override {
    Var v = GetRef<Var>(op);
    if (uf_.Contains(v)) {
      // visit the result in case the variable is remapped
      return ExprMutator::VisitExpr_(uf_.Lookup(v).as<VarNode>());
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const DataflowVarNode* op) override {
    Var v = GetRef<Var>(op);
    if (uf_.Contains(v)) {
      // visit in case of variable remapping
      return ExprMutator::VisitExpr_(uf_.Lookup(v).as<DataflowVarNode>());
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const FunctionNode* op) override {
    // add function parameters into the current scope before proceeding
    for (auto v : op->params) {
      uf_.InsertAnchor(v);
    }
    return ExprMutator::VisitExpr_(op);
  }

  // populate the union-find before processing the bindings
  Expr VisitExpr_(const SeqExprNode* op) override {
    for (auto bb : op->blocks) {
      for (auto binding : bb->bindings) {
        ProcessBinding(binding);
      }
    }
    return ExprMutator::VisitExpr_(op);
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
    // for match shape, we need to be cleverer and allow the shape_ to change
    // due to possible substitutions
    Expr new_value = this->VisitExpr(binding->value);
    Expr new_pattern = this->VisitExpr(ShapeExpr(binding->pattern));

    Var new_var;
    if (binding->var.defined()) {
      Optional<Expr> new_shape;
      if (new_value->checked_type_.defined() && new_value->checked_type_.as<DynTensorTypeNode>()) {
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
  // adds vars to the union find if eligible
  void ProcessBinding(const Binding& binding) {
    if (const VarBindingNode* vb = binding.as<VarBindingNode>()) {
      UpdateDef(vb->var, vb->value);
      return;
    }
    const MatchShapeNode* m = binding.as<MatchShapeNode>();
    CHECK(m);
    if (m->var.defined()) {
      UpdateDef(m->var, m->value);
    }
  }

  // if the RHS (value) is a var and eligible to unify,
  // insert the binding into the union-find;
  // otherwise, add the newly defined var as an anchor
  void UpdateDef(const Var& def, const Expr& value) {
    const VarNode* v = value.as<VarNode>();
    if (v && CanUnifyVars(def, GetRef<Var>(v))) {
      uf_.InsertBinding(def, GetRef<Var>(v));
      return;
    }
    uf_.InsertAnchor(def);
  }

  bool AnnotationsDiffer(const ObjectRef& obj1, const ObjectRef& obj2,
                         std::function<bool(const ObjectRef&, const ObjectRef&)> check_eq) {
    // annotations differ if one is present but not the other
    // or they're both present and they differ structurally
    bool both_present = obj1.defined() && obj2.defined();
    bool neither_present = !obj1.defined() && !obj2.defined();
    return !(both_present || neither_present) || (both_present && !check_eq(obj1, obj2));
  }

  bool CanUnifyVars(Var lhs, Var rhs) {
    // Cases when we conservatively do not unify:
    // 1. checked_type_ or shape_ of the LHS differ from that of the RHS.
    //    In this case, we could be overriding user annotations.
    // 2. If the LHS is a Var and the parent of the RHS is a DataflowVar.
    //    That could result in a DataflowVar leaving the current DataflowBlock.
    Var parent = uf_.Lookup(rhs);
    bool annotations_differ =
        AnnotationsDiffer(lhs->shape_, parent->shape_,
                          [&](auto shape1, auto shape2) {
                            return builder_->CanProveShapeEqual(Downcast<Expr>(shape1),
                                                                Downcast<Expr>(shape2));
                          }) ||
        AnnotationsDiffer(lhs->checked_type_, parent->checked_type_, [&](auto type1, auto type2) {
          return tvm::StructuralEqual()(type1, type2);
        });
    bool var_to_dataflow = (!lhs.as<DataflowVarNode>() && parent.as<DataflowVarNode>());
    return !annotations_differ && !var_to_dataflow;
  }

  VarUnifier uf_;
};

Expr Canonicalize(const Expr& e) { return Canonicalizer().VisitExpr(e); }

namespace transform {

Pass Canonicalize() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(Canonicalize(f)); };
  return CreateFunctionPass(pass_func, 1, "Canonicalize", {});
}

TVM_REGISTER_GLOBAL("relax.transform.Canonicalize").set_body_typed(Canonicalize);

}  // namespace transform

}  // namespace relax
}  // namespace tvm