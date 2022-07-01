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
 * \file src/relax/type.cc
 * \brief Relax's type system AST nodes throughout the IR.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(ShapeTypeNode);

ShapeType::ShapeType(Span span) {
  ObjectPtr<ShapeTypeNode> n = make_object<ShapeTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ShapeType").set_body_typed([](Span span) { return ShapeType(span); });

ObjectType::ObjectType(Span span) {
  ObjectPtr<ObjectTypeNode> n = make_object<ObjectTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ObjectTypeNode);

TVM_REGISTER_GLOBAL("relax.ObjectType").set_body_typed([](Span span) { return ObjectType(span); });

DynTensorType::DynTensorType(int ndim, DataType dtype, Span span) {
  ObjectPtr<DynTensorTypeNode> n = make_object<DynTensorTypeNode>();
  n->ndim = std::move(ndim);
  n->dtype = std::move(dtype);
  n->span = span;
  data_ = std::move(n);
}

DynTensorType DynTensorType::CreateUnknownNDim(DataType dtype, Span span) {
  ObjectPtr<DynTensorTypeNode> n = make_object<DynTensorTypeNode>();
  n->ndim = -1;
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  return DynTensorType(std::move(n));
}

TVM_REGISTER_NODE_TYPE(DynTensorTypeNode);

TVM_REGISTER_GLOBAL("relax.DynTensorType").set_body_typed([](int ndim, DataType dtype, Span span) {
  return DynTensorType(ndim, dtype, span);
});

DimType::DimType(Span span) {
  ObjectPtr<DimTypeNode> n = make_object<DimTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DimTypeNode);

TVM_REGISTER_GLOBAL("relax.DimType").set_body_typed([](Span span) { return DimType(span); });

/*!
 * \brief Utility class for generic type dispatching:
 * VisitType dispatches on the base type and checks if the derived type is a subtype of the base
 * type.
 */
class BaseTypeChecker : public TypeFunctor<bool(const Type& n)> {
 public:
  explicit BaseTypeChecker(const Type& derived) : derived_{derived} {}

  bool VisitType_(const ShapeTypeNode* base) final {
    if (derived_.as<ShapeTypeNode>()) {
      return true;
    }
    return false;
  }
  bool VisitType_(const ObjectTypeNode* base) final { return true; }

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

}  // namespace relax
}  // namespace tvm
