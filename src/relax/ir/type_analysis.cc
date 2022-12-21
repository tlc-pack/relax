
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
 * \file src/relax/ir/type_analysis.cc
 * \brief Relax type analysis APIs.
 */

#include <tvm/ir/type_functor.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_analysis.h>

namespace tvm {
namespace relax {
/*!
 * \brief Utility class for generic type dispatching:
 * VisitType dispatches on the base type and checks if the derived type is a subtype of the base
 * type.
 */
class BaseTypeChecker : public TypeFunctor<bool(const Type& n)> {
 public:
  explicit BaseTypeChecker(const Type& derived) : derived_{derived} {}

  bool VisitType_(const ObjectTypeNode* base) final { return true; }

  bool VisitType_(const ShapeTypeNode* base) final {
    if (auto* rhs = derived_.as<ShapeTypeNode>()) {
      return base->ndim == kUnknownDim || base->ndim == rhs->ndim;
    }
    return false;
  }

  bool VisitType_(const DynTensorTypeNode* base) final {
    if (auto derived_tensor = derived_.as<DynTensorTypeNode>()) {
      if (base->IsUnknownNdim() || base->ndim == derived_tensor->ndim) {
        if (base->IsUnknownDtype() || base->dtype == derived_tensor->dtype) {
          return true;
        }
      }
    }
    return false;
  }

  bool VisitType_(const TupleTypeNode* base) final {
    if (auto derived_tuple = derived_.as<TupleTypeNode>()) {
      if (base->fields.size() != derived_tuple->fields.size()) {
        return false;
      }

      for (size_t i = 0; i < base->fields.size(); ++i) {
        if (!IsBaseOf(base->fields[i], derived_tuple->fields[i])) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  bool VisitType_(const FuncTypeNode* base) final {
    if (auto derived_func = derived_.as<FuncTypeNode>()) {
      if (base->arg_types.size() != derived_func->arg_types.size()) {
        return false;
      }
      for (size_t i = 0; i < base->arg_types.size(); ++i) {
        if (!IsBaseOf(base->arg_types[i], derived_func->arg_types[i])) {
          return false;
        }
      }
      if (!IsBaseOf(base->ret_type, derived_func->ret_type)) {
        return false;
      }
      return true;
    }
    return false;
  }

  bool VisitType_(const PackedFuncTypeNode* base) final {
    if (derived_.as<PackedFuncTypeNode>()) {
      return true;
    }
    return false;
  }

 private:
  Type derived_;
};

bool IsBaseOf(const Type& base, const Type& derived) {
  BaseTypeChecker visitor(derived);
  return visitor.VisitType(base);
}

TVM_REGISTER_GLOBAL("relax.IsBaseOf").set_body_typed([](const Type& base, const Type& derived) {
  return IsBaseOf(base, derived);
});

/*!
 * \brief Utility class for finding the LCA (lowest common ancestor) of two types.
 */
class LCAVisitor : public TypeFunctor<Type(const Type& n)> {
 public:
  explicit LCAVisitor(const Type& u) : u_{u} {}

  Type VisitType_(const ObjectTypeNode* t) final { return ObjectType(); }

  Type VisitType_(const ShapeTypeNode* t) final {
    if (auto* rhs = u_.as<ShapeTypeNode>()) {
      if (t->ndim == rhs->ndim) return GetRef<ShapeType>(t);
      return ShapeType();
    }
    return ObjectType();
  }

  Type VisitType_(const DynTensorTypeNode* t) final {
    if (auto u_tensor = u_.as<DynTensorTypeNode>()) {
      int res_ndim = t->ndim;
      DataType res_dtype = t->dtype;
      if (t->ndim != u_tensor->ndim) {
        res_ndim = -1;
      }
      if (t->dtype != u_tensor->dtype) {
        res_dtype = DataType::Void();
      }
      return DynTensorType(res_ndim, res_dtype);
    }
    return ObjectType();
  }

  Type VisitType_(const TupleTypeNode* t) final {
    if (auto u_tuple = u_.as<TupleTypeNode>()) {
      if (t->fields.size() != u_tuple->fields.size()) {
        return ObjectType();
      }
      Array<Type> res_fields_types;
      for (size_t i = 0; i < t->fields.size(); ++i) {
        res_fields_types.push_back(FindLCA(t->fields[i], u_tuple->fields[i]));
      }
      return TupleType(res_fields_types);
    }
    return ObjectType();
  }

  Type VisitType_(const FuncTypeNode* t) final {
    if (auto u_func = u_.as<FuncTypeNode>()) {
      if (t->arg_types.size() != u_func->arg_types.size()) {
        return ObjectType();
      }
      Array<Type> res_arg_types;
      for (size_t i = 0; i < t->arg_types.size(); ++i) {
        // TODO(yuchen): figure out what the result Function's arg_types should be if
        // fields don't match
        ICHECK(StructuralEqual()(t->arg_types[i], u_func->arg_types[i]))
            << "The two functions' arg_types do not match.";
        res_arg_types.push_back(FindLCA(t->arg_types[i], u_func->arg_types[i]));
      }
      return FuncType(res_arg_types, FindLCA(t->ret_type, u_func->ret_type), {}, {});
    }
    return ObjectType();
  }

  Type VisitType_(const PackedFuncTypeNode* t) final {
    if (u_.as<PackedFuncTypeNode>()) {
      return PackedFuncType();
    }
    return ObjectType();
  }

 private:
  Type u_;
};

Type FindLCA(const Type& t, const Type& u) {
  LCAVisitor visitor(u);
  return visitor.VisitType(t);
}

TVM_REGISTER_GLOBAL("relax.FindLCA").set_body_typed([](const Type& t, const Type& u) {
  return FindLCA(t, u);
});

}  // namespace relax
}  // namespace tvm
