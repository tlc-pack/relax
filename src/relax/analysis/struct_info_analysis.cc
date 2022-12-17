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
 * \file struct_info_analysis.cc
 * \brief Implementations of foundation struct info analysis
 *
 * \note Update this file when you added a new StructInfo.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace relax {

//--------------------------
// GetStaticType
//--------------------------
class StaticTypeDeriver : public StructInfoFunctor<Type(const StructInfo&)> {
 public:
  Type VisitStructInfo_(const ObjectStructInfoNode* op) final { return ObjectType(op->span); }

  Type VisitStructInfo_(const PrimStructInfoNode* op) final {
    return PrimType(op->dtype, op->span);
  }

  Type VisitStructInfo_(const ShapeStructInfoNode* op) final {
    return ShapeType(op->ndim, op->span);
  }

  Type VisitStructInfo_(const TensorStructInfoNode* op) final {
    return DynTensorType(op->ndim, op->dtype);
  }

  Type VisitStructInfo_(const TupleStructInfoNode* op) final {
    Array<Type> fields =
        op->fields.Map([this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });
    return TupleType(fields, op->span);
  }

  Type VisitStructInfo_(const FuncStructInfoNode* op) final {
    if (op->IsOpaque()) return PackedFuncType(op->span);
    Array<Type> params = op->params.value().Map(
        [this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });
    Type ret = this->VisitStructInfo(op->ret);
    return FuncType(params, ret, {}, {}, op->span);
  }
};

Type GetStaticType(const StructInfo& info) { return StaticTypeDeriver()(info); }

TVM_REGISTER_GLOBAL("relax.analysis.GetStaticType").set_body_typed([](const StructInfo& info) {
  return GetStaticType(info);
});

//--------------------------
// GetLegacyShapeHint
//--------------------------
// TODO(relax-team) remove this function after phasing out shape.
class LegacyShapeDeriver : public StructInfoFunctor<Optional<Expr>(const StructInfo&)> {
 public:
  Optional<Expr> VisitStructInfo_(const ObjectStructInfoNode* op) final { return NullOpt; }

  Optional<Expr> VisitStructInfo_(const PrimStructInfoNode* op) final { return NullOpt; }

  Optional<Expr> VisitStructInfo_(const ShapeStructInfoNode* op) final { return NullOpt; }

  Optional<Expr> VisitStructInfo_(const TensorStructInfoNode* op) final {
    if (op->shape.defined()) {
      return op->shape;
    } else {
      return RuntimeDepShape();
    }
  }

  Optional<Expr> VisitStructInfo_(const TupleStructInfoNode* op) final {
    bool valid = true;
    Array<Expr> fields = op->fields.Map([this, &valid](const StructInfo& sinfo) {
      Optional<Expr> shape = this->VisitStructInfo(sinfo);
      valid &= shape.defined();
      return shape.value_or(Expr(nullptr));
    });

    // recursively collect structinfo to make sure legacy shape is also well defined.
    if (valid && fields.size() != 0) {
      Tuple tuple(fields, op->span);
      Array<StructInfo> tuple_sinfo;
      for (Expr field : tuple->fields) {
        tuple_sinfo.push_back(GetStructInfo(field));
      }
      UpdateStructInfo(tuple, TupleStructInfo(tuple_sinfo));
      return tuple;
    } else {
      return NullOpt;
    }
  }

  Optional<Expr> VisitStructInfo_(const FuncStructInfoNode* op) final { return NullOpt; }
};

Optional<Expr> GetLegacyShapeHint(const StructInfo& info) { return LegacyShapeDeriver()(info); }

TVM_REGISTER_GLOBAL("relax.analysis.GetLegacyShapeHint").set_body_typed([](const StructInfo& info) {
  return GetLegacyShapeHint(info);
});

//--------------------------
// StructInfoFromType
//--------------------------

StructInfo StructInfoFromType(const Type& type) {
  return StructInfoFromTypeLegacyShapeHint(type, NullOpt);
}

StructInfo StructInfoFromTypeLegacyShapeHint(const Type& type, Optional<Expr> shape_hint) {
  if (type.as<ObjectTypeNode>()) {
    return ObjectStructInfo(type->span);
  } else if (const PrimTypeNode* prim_type = type.as<PrimTypeNode>()) {
    return PrimStructInfo(prim_type->dtype, prim_type->span);
  } else if (const ShapeTypeNode* shape_type = type.as<ShapeTypeNode>()) {
    return ShapeStructInfo(shape_type->ndim, type->span);
  } else if (const DynTensorTypeNode* tensor_type = type.as<DynTensorTypeNode>()) {
    if (!shape_hint.defined() || shape_hint->IsInstance<RuntimeDepShapeNode>()) {
      return TensorStructInfo(tensor_type->dtype, tensor_type->ndim);
    } else {
      return TensorStructInfo(shape_hint.value(), tensor_type->dtype);
    }
  } else if (const TupleTypeNode* tuple_type = type.as<TupleTypeNode>()) {
    Array<StructInfo> fields;
    if (shape_hint.defined() && shape_hint.value()->IsInstance<TupleNode>()) {
      Array<Expr> shape_hint_fields = Downcast<Tuple>(shape_hint.value())->fields;
      ICHECK_EQ(shape_hint_fields.size(), tuple_type->fields.size());
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        fields.push_back(
            StructInfoFromTypeLegacyShapeHint(tuple_type->fields[i], shape_hint_fields[i]));
      }
    } else {
      for (const Type& field : tuple_type->fields) {
        fields.push_back(StructInfoFromType(field));
      }
    }
    return TupleStructInfo(fields, type->span);
  } else if (const FuncTypeNode* func_type = type.as<FuncTypeNode>()) {
    Array<StructInfo> params =
        func_type->arg_types.Map([](const Type& param) { return StructInfoFromType(param); });
    StructInfo ret = StructInfoFromType(func_type->ret_type);
    return FuncStructInfo(params, ret, func_type->span);
  } else {
    LOG(FATAL) << "Unsupported type: " << type;
    return StructInfo();
  }
}
//--------------------------
// EraseToWellDefined
//--------------------------
class WellDefinedEraser : public StructInfoMutator,
                          public ExprMutatorBase,
                          public tir::ExprMutator {
 public:
  WellDefinedEraser(std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map,
                    std::function<Optional<Expr>(const Var& var)> f_var_map, arith::Analyzer* ana)
      : f_shape_var_map_(f_shape_var_map), f_var_map_(f_var_map), ana_(ana) {}

  StructInfo VisitStructInfo_(const ShapeStructInfoNode* op) final {
    bool has_undefined = false;
    Optional<Array<PrimExpr>> values;

    if (op->values.defined()) {
      std::swap(has_undefined_, has_undefined);
      values = op->values.value().Map([&](PrimExpr val) { return this->VisitPrimExpr(val); });
      std::swap(has_undefined_, has_undefined);
    }
    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (values.same_as(op->values)) {
        return GetRef<StructInfo>(op);
      } else {
        return ShapeStructInfo(values.value(), op->span);
      }
    } else {
      return ShapeStructInfo(op->ndim, op->span);
    }
  }

  StructInfo VisitStructInfo_(const TensorStructInfoNode* op) final {
    bool has_undefined = false;
    Optional<Expr> shape;

    if (op->shape.defined()) {
      std::swap(has_undefined_, has_undefined);
      shape = relax::ExprMutatorBase::VisitExpr(op->shape.value());
      std::swap(has_undefined_, has_undefined);
    }

    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (shape.same_as(op->shape)) {
        return GetRef<StructInfo>(op);
      } else {
        if (shape.defined()) {
          return TensorStructInfo(shape.value(), op->dtype, op->span);
        } else {
          return TensorStructInfo(op->dtype, op->ndim, op->span);
        }
      }
    } else {
      return TensorStructInfo(op->dtype, op->ndim, op->span);
    }
  }

  StructInfo VisitStructInfo_(const FuncStructInfoNode* op) final {
    // NOTE: we always require func struct info to be well formed.
    // and need to avoid recursing into it as the vars are defined in params.
    return GetRef<StructInfo>(op);
  }

  using relax::ExprMutatorBase::VisitExpr_;
  using tir::ExprMutator::VisitExpr_;

  // connect things up
  PrimExpr VisitPrimExpr(const PrimExpr& expr) {
    // apply eager simplification
    PrimExpr val = tir::ExprMutator::VisitExpr(expr);
    if (!val.same_as(expr)) {
      return ana_->Simplify(val);
    } else {
      return val;
    }
  }

  Expr VisitExpr_(const DataflowVarNode* var) final {
    return VisitExpr_(static_cast<const VarNode*>(var));
  }

  Expr VisitExpr_(const VarNode* var) final {
    Optional<Expr> ret;
    if (f_var_map_ != nullptr) {
      ret = f_var_map_(GetRef<Var>(var));
    }
    has_undefined_ = has_undefined_ || !ret.defined();
    if (ret.defined()) {
      ICHECK(ret.as<VarNode>() || ret.as<ShapeExprNode>())
          << "Only allow Expr in StructInfo to be ShapeExpr or Var";
    }
    return ret.value_or(GetRef<Expr>(var));
  }

  PrimExpr VisitExpr_(const tir::VarNode* var) final {
    Optional<PrimExpr> ret;
    if (f_shape_var_map_ != nullptr) {
      ret = f_shape_var_map_(GetRef<tir::Var>(var));
    }
    has_undefined_ = has_undefined_ || !ret.defined();

    if (ret.defined()) {
      PrimExpr value = ret.value();
      if (value->IsInstance<IntImmNode>()) {
        return tvm::cast(DataType::Int(64), value);
      }
      ICHECK(value.dtype() == DataType::Int(64)) << "Can only provide i64 expressions in shape";
      return value;
    } else {
      return GetRef<PrimExpr>(var);
    }
  }

 private:
  bool has_undefined_ = false;
  std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map_;
  std::function<Optional<Expr>(const Var& var)> f_var_map_;
  arith::Analyzer* ana_;
};

StructInfo EraseToWellDefined(
    const StructInfo& info, std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map,
    std::function<Optional<Expr>(const Var& var)> f_var_map, arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return WellDefinedEraser(f_shape_var_map, f_var_map, &inst).VisitStructInfo(info);
  } else {
    return WellDefinedEraser(f_shape_var_map, f_var_map, ana).VisitStructInfo(info);
  }
}

TVM_REGISTER_GLOBAL("relax.analysis.EraseToWellDefined")
    .set_body_typed([](const StructInfo& info, Map<tir::Var, PrimExpr> shape_var_map,
                       Map<Var, Expr> var_map) {
      std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map = nullptr;
      std::function<Optional<Expr>(const Var& var)> f_var_map = nullptr;

      if (!shape_var_map.empty()) {
        f_shape_var_map = [&](const tir::Var& var) -> Optional<PrimExpr> {
          auto it = shape_var_map.find(var);
          if (it != shape_var_map.end()) return (*it).second;
          return NullOpt;
        };
      }

      if (!var_map.empty()) {
        f_var_map = [&](const Var& var) -> Optional<Expr> {
          auto it = var_map.find(var);
          if (it != var_map.end()) return (*it).second;
          return NullOpt;
        };
      }

      return EraseToWellDefined(info, f_shape_var_map, f_var_map);
    });

//--------------------------
// IsBaseOf
//--------------------------
class StructInfoBaseChecker : public StructInfoFunctor<bool(const StructInfo&, const StructInfo&)> {
 public:
  explicit StructInfoBaseChecker(arith::Analyzer* ana) : analyzer_(ana) {}

  bool VisitStructInfo(const StructInfo& lhs, const StructInfo& other) final {
    // quick path
    if (lhs.same_as(other)) return true;
    return StructInfoFunctor::VisitStructInfo(lhs, other);
  }

  // Object is based of everything
  bool VisitStructInfo_(const ObjectStructInfoNode* lhs, const StructInfo& other) final {
    return true;
  }

  bool VisitStructInfo_(const PrimStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<PrimStructInfoNode>();
    if (rhs == nullptr) return false;
    return lhs->dtype == rhs->dtype;
  }

  bool VisitStructInfo_(const ShapeStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<ShapeStructInfoNode>();
    if (rhs == nullptr) return false;
    // lhs have unknown ndim
    if (lhs->IsUnknownNdim()) return true;

    // ndim must match
    if (lhs->ndim != rhs->ndim) return false;

    // lhs does not have symbolic value
    if (!lhs->values.defined()) return true;

    // rhs does not have symbolic value but lhs do.
    if (!rhs->values.defined()) return false;
    if (!CanProveShapeEqual(lhs->values.value(), rhs->values.value(), analyzer_)) return false;
    return true;
  }

  bool VisitStructInfo_(const TensorStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TensorStructInfoNode>();
    if (rhs == nullptr) return false;
    // dtype mismatch
    if (!lhs->IsUnknownDtype() && lhs->dtype != rhs->dtype) return false;
    // ndim mismatch
    if (!lhs->IsUnknownNdim() && lhs->ndim != rhs->ndim) return false;
    // lhs does not have defined shape and everything else matches
    if (!lhs->shape.defined()) return true;
    // rhs does not have symbolic value but lhs doesn't
    if (!rhs->shape.defined()) return false;
    // shape match check
    if (!CanProveShapeEqual(lhs->shape.value(), rhs->shape.value(), analyzer_)) return false;
    return true;
  }

  bool VisitStructInfo_(const TupleStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TupleStructInfoNode>();
    if (rhs == nullptr) return false;
    return BaseCheckArray(lhs->fields, rhs->fields);
  }

  bool VisitStructInfo_(const FuncStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<FuncStructInfoNode>();
    if (rhs == nullptr) return false;

    // lhs opaque handling
    if (lhs->IsOpaque()) {
      if (lhs->derive_func.defined()) {
        return lhs->derive_func.same_as(rhs->derive_func);
      }
      // no derivation function, only depends on ret
      return this->VisitStructInfo(lhs->ret, rhs->ret);
    }

    // rhs is opaque but lhs is not
    if (rhs->IsOpaque()) return false;

    // NOTE: lhs->params, rhs->params may contain different symbolic
    // vars that needs to be re-mapped to each other.
    // This can only be done through structural equality check and not BaseCheckArray.
    //
    // So we check structural equality here and if two are structurally
    // equal return true.
    //
    // otherwise we do best effort BaseArrayCheck.
    //
    // This still does not handle cases where some arguments are sub of another
    // while other parameters needs to get remapped.
    //
    // Given we only do best effort checking in these cases, and such cases
    // are likely not a primary concern atm, we take this approach here.
    if (struct_equal_(GetRef<StructInfo>(lhs), other)) return true;

    // general function
    if (!BaseCheckArray(lhs->params.value(), rhs->params.value())) {
      return false;
    }
    if (!this->VisitStructInfo(lhs->ret, rhs->ret)) {
      return false;
    }
    return true;
  }

 private:
  // analyzer
  arith::Analyzer* analyzer_;
  // struct equal checker
  StructuralEqual struct_equal_;

  // check arrays
  bool BaseCheckArray(const Array<StructInfo>& lhs, const Array<StructInfo>& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!this->VisitStructInfo(lhs[i], rhs[i])) return false;
    }
    return true;
  }
};

bool IsBaseOf(const StructInfo& base, const StructInfo& derived, arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return StructInfoBaseChecker(&inst)(base, derived);
  } else {
    return StructInfoBaseChecker(ana)(base, derived);
  }
}

TVM_REGISTER_GLOBAL("relax.StructInfoIsBaseOf")
    .set_body_typed([](const StructInfo& base, const StructInfo& derived) {
      return IsBaseOf(base, derived);
    });

//--------------------------
// UnifyToLCA
//--------------------------
class StructInfoLCAFinder
    : public StructInfoFunctor<StructInfo(const StructInfo&, const StructInfo&)> {
 public:
  explicit StructInfoLCAFinder(arith::Analyzer* ana) : analyzer_(ana) {}

  StructInfo VisitStructInfo(const StructInfo& lhs, const StructInfo& other) final {
    // quick path
    if (lhs.same_as(other)) return lhs;
    return StructInfoFunctor::VisitStructInfo(lhs, other);
  }

  // Object is based of everything, unify to object.
  StructInfo VisitStructInfo_(const ObjectStructInfoNode* lhs, const StructInfo& other) final {
    return GetRef<StructInfo>(lhs);
  }

  StructInfo VisitStructInfo_(const PrimStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<PrimStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);
    if (lhs->dtype == rhs->dtype) return GetRef<StructInfo>(lhs);
    return ObjectStructInfo(lhs->span);
  }

  StructInfo VisitStructInfo_(const ShapeStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<ShapeStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);

    int ndim = lhs->ndim == rhs->ndim ? lhs->ndim : kUnknownDim;
    if (lhs->ndim != rhs->ndim || !lhs->values.defined() || !rhs->values.defined() ||
        !CanProveShapeEqual(lhs->values.value(), rhs->values.value(), analyzer_)) {
      // prefers return same when possible
      if (!lhs->values.defined() && lhs->ndim == ndim) {
        return GetRef<StructInfo>(lhs);
      } else {
        return ShapeStructInfo(ndim, lhs->span);
      }
    }
    // equals to each other
    return GetRef<StructInfo>(lhs);
  }

  StructInfo VisitStructInfo_(const TensorStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TensorStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);

    // find the target dtype and ndim.
    DataType dtype = lhs->dtype == rhs->dtype ? lhs->dtype : DataType::Void();
    int ndim = lhs->ndim == rhs->ndim ? lhs->ndim : kUnknownDim;
    // if ndim mismatch or one side of shape is missing
    // then we cannot keep in symbolic shape
    if (lhs->ndim != rhs->ndim || !lhs->shape.defined() || !rhs->shape.defined() ||
        !CanProveShapeEqual(lhs->shape.value(), rhs->shape.value(), analyzer_)) {
      // reuse lhs when possible
      if (!lhs->shape.defined() && lhs->dtype == dtype && lhs->ndim == ndim) {
        return GetRef<StructInfo>(lhs);
      } else {
        return TensorStructInfo(dtype, ndim, lhs->span);
      }
    }
    // symbolic shape match but dtype mismatch
    if (lhs->dtype != dtype) {
      return TensorStructInfo(lhs->shape.value(), dtype, lhs->span);
    } else {
      return GetRef<StructInfo>(lhs);
    }
  }

  StructInfo VisitStructInfo_(const TupleStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TupleStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);
    Optional<Array<StructInfo>> fields = UnifyArray(lhs->fields, rhs->fields);
    // tuple length not the same.
    if (!fields.defined()) return ObjectStructInfo(lhs->span);

    // same length tuple.
    if (!fields.same_as(lhs->fields)) {
      return TupleStructInfo(fields.value(), lhs->span);
    } else {
      return GetRef<StructInfo>(lhs);
    }
  }

  StructInfo VisitStructInfo_(const FuncStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<FuncStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);

    // lhs opaque handling
    if (lhs->IsOpaque()) {
      if (lhs->derive_func.defined()) {
        if (lhs->derive_func.same_as(rhs->derive_func)) {
          return GetRef<StructInfo>(lhs);
        } else {
          // Create a new opaque with object return
          return FuncStructInfo::OpaqueFunc(ObjectStructInfo(), lhs->span);
        }
      } else {
        // no derivation function, only depends on ret
        StructInfo ret = this->VisitStructInfo(lhs->ret, rhs->ret);
        if (ret.same_as(lhs->ret)) return GetRef<StructInfo>(lhs);
        return FuncStructInfo::OpaqueFunc(ret, lhs->span);
      }
    }
    // rhs is opaque, lhs is not
    if (rhs->IsOpaque()) {
      // unify ret value, note that rhs's ret is context free(because it is opaque)
      // so result of the unify is also context-free.
      StructInfo ret = this->VisitStructInfo(lhs->ret, rhs->ret);
      return FuncStructInfo::OpaqueFunc(ret, lhs->span);
    }

    // Both lhs and rhs are not opaque
    // NOTE: lhs->params, rhs->params may contain different symbolic
    // vars that needs to be re-mapped to each other.
    // This can only be done through structural equality check.
    //
    // So we check structural equality here and if two are structurally
    // equal return true.
    //
    // otherwise we do best effort of unify types without considering var remap.
    //
    // This still does not handle cases where some arguments are sub of another
    // while other parameters needs to get remapped.
    //
    // Given we only do best effort checking in these cases, and such cases
    // are likely not a primary concern atm, we take this approach here.
    if (struct_equal_(GetRef<StructInfo>(lhs), GetRef<StructInfo>(rhs))) {
      return GetRef<StructInfo>(lhs);
    }

    auto params = UnifyArray(lhs->params.value(), rhs->params.value());
    auto ret = this->VisitStructInfo(lhs->ret, rhs->ret);

    if (params.same_as(lhs->params) && ret.same_as(lhs->ret)) {
      return GetRef<StructInfo>(lhs);
    } else {
      // fail to unify the params
      if (!params.defined()) {
        return FuncStructInfo::OpaqueFunc(ret, lhs->span);
      } else {
        return FuncStructInfo(params.value(), ret, lhs->span);
      }
    }
  }

 private:
  // analyzer
  arith::Analyzer* analyzer_;
  // struct equal checker
  StructuralEqual struct_equal_;

  // check arrays
  Optional<Array<StructInfo>> UnifyArray(const Array<StructInfo>& lhs,
                                         const Array<StructInfo>& rhs) {
    if (lhs.same_as(rhs)) return lhs;
    if (lhs.size() != rhs.size()) return NullOpt;
    size_t index = 0;
    return lhs.Map([&](const StructInfo& a) { return this->VisitStructInfo(a, rhs[index++]); });
  }
};

StructInfo StructInfoLCA(const StructInfo& lhs, const StructInfo& rhs, arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return StructInfoLCAFinder(&inst)(lhs, rhs);
  } else {
    return StructInfoLCAFinder(ana)(lhs, rhs);
  }
}

TVM_REGISTER_GLOBAL("relax.analysis.StructInfoLCA")
    .set_body_typed([](const StructInfo& lhs, const StructInfo& rhs) {
      return StructInfoLCA(lhs, rhs);
    });

}  // namespace relax
}  // namespace tvm
