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
 *
 * \file analysis.cc
 *
 * \brief Utility functions for Relax.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>

namespace tvm {
namespace relax {

template <typename T>
struct InsertionSet {
  std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual> set;
  std::vector<T> data;
  void Insert(const T& t) {
    if (set.count(t) == 0) {
      set.insert(t);
      data.push_back(t);
    }
  }
};

class TypeVarEVisitor : private ExprVisitor {
 public:
  explicit TypeVarEVisitor(const IRModule& mod) : mod_(mod) {}

  Array<TypeVar> CollectFree() {
    Array<TypeVar> ret;
    for (const auto& v : type_vars_.data) {
      if (bound_type_vars_.set.count(v) == 0) {
        ret.push_back(v);
      }
    }
    return ret;
  }

  Array<TypeVar> CollectBound() {
    Array<TypeVar> ret;
    for (const auto& v : bound_type_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<TypeVar> CollectAll() {
    Array<TypeVar> ret;
    for (const auto& v : type_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<TypeVar> Free(const Expr& expr) {
    VisitExpr(expr);
    return CollectFree();
  }

  Array<TypeVar> Free(const Type& type) {
    VisitType(type);
    return CollectFree();
  }

  Array<TypeVar> Bound(const Expr& expr) {
    VisitExpr(expr);
    return CollectBound();
  }

  Array<TypeVar> Bound(const Type& type) {
    VisitType(type);
    return CollectBound();
  }

  Array<TypeVar> All(const Expr& expr) {
    VisitExpr(expr);
    return CollectAll();
  }

  Array<TypeVar> All(const Type& type) {
    VisitType(type);
    return CollectAll();
  }

  void VisitExpr_(const FunctionNode* f) final {
    /*
    for (const auto& tp : f->type_params) {
      type_vars_.Insert(tp);
      bound_type_vars_.Insert(tp);
    }
    */
    ExprVisitor::VisitExpr_(f);
  }

 private:
  InsertionSet<TypeVar> type_vars_;
  InsertionSet<TypeVar> bound_type_vars_;
  const IRModule& mod_;
};

class VarVisitor : protected ExprVisitor {
 public:
  Array<Var> Free(const Expr& expr) {
    this->VisitExpr(expr);
    Array<Var> ret;
    for (const auto& v : vars_.data) {
      if (bound_vars_.set.count(v) == 0) {
        ret.push_back(v);
      }
    }
    return ret;
  }

  Array<Var> Collect() {
    Array<Var> ret;
    for (const auto& v : bound_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<Var> Bound(const Expr& expr) {
    this->VisitExpr(expr);
    return Collect();
  }

  Array<Var> All(const Expr& expr) {
    this->VisitExpr(expr);
    Array<Var> ret;
    for (const auto& v : vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  void MarkBounded(const Var& v) {
    bound_vars_.Insert(v);
    vars_.Insert(v);
  }

  void VisitExpr_(const VarNode* var) final { vars_.Insert(GetRef<Var>(var)); }

  void VisitExpr_(const FunctionNode* op) final {
    for (const auto& param : op->params) {
      MarkBounded(param);
    }
    VisitExpr(op->body);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    MarkBounded(binding->var);
    VisitExpr(binding->value);
    VisitVarDef(binding->var);
  }

 private:
  InsertionSet<Var> vars_;
  InsertionSet<Var> bound_vars_;
};

tvm::Array<Var> FreeVars(const Expr& expr) { return VarVisitor().Free(expr); }

tvm::Array<Var> BoundVars(const Expr& expr) { return VarVisitor().Bound(expr); }

tvm::Array<Var> AllVars(const Expr& expr) { return VarVisitor().All(expr); }

TVM_REGISTER_GLOBAL("relax.analysis.free_vars").set_body_typed(FreeVars);

TVM_REGISTER_GLOBAL("relax.analysis.bound_vars").set_body([](TVMArgs args, TVMRetValue* ret) {
  ObjectRef x = args[0];
  if (x.as<ExprNode>()) {
    *ret = BoundVars(Downcast<Expr>(x));
  }
});

TVM_REGISTER_GLOBAL("relax.analysis.all_vars").set_body_typed(AllVars);

}  // namespace relax
}  // namespace tvm
