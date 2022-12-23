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
 * \file src/relax/backend/vm/vm_shape_lower.cc
 * \brief Lower the shape expressions in relax to VM shape heap manipulations and generate related
 * TIR functions to do shape calculations.
 */
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

class PrimExprSlotCollector : public ExprVisitor, public StructInfoVisitor {
 public:
  // collect the PrimExpr slot for a given function
  static Map<PrimExpr, Integer> Collect(Function func) {
    PrimExprSlotCollector collector;
    // collect shape delcarations in func params
    for (auto param : func->params) {
      collector.VisitStructInfo(GetStructInfo(param));
      collector.VisitExpr(param);
    }
    collector.VisitExpr(func->body);
    // avoid create any slot for static shape.
    if (!collector.dyn_shape_) collector.slot_map_.clear();
    return std::move(collector.slot_map_);
  }

 private:
  void VisitPrimExpr(const PrimExpr& expr) final {
    if (!expr->IsInstance<IntImmNode>()) {
      dyn_shape_ = true;
    }
    if (slot_map_.count(expr) == 0) {
      slot_map_.Set(expr, slot_count_++);
    }
  }

  void VisitBinding_(const MatchCastNode* op) final {
    // Visit the match cast struct info so we can define
    // the symbolic variables here.
    this->VisitStructInfo(op->struct_info);
  }

  void VisitExpr_(const FunctionNode* op) final {
    // Do not recurse into function node as it is self-contained
  }

  void VisitStructInfo_(const FuncStructInfoNode* op) final {
    // Do not recurse into function struct info as it is self-contained
  }

  void VisitStructInfoExprField(const PrimExpr& expr) final { VisitPrimExpr(expr); }

  void VisitStructInfoExprField(const Expr& expr) final { ExprVisitor::VisitExpr(expr); }

  bool dyn_shape_ = false;
  int slot_count_ = 0;
  Map<PrimExpr, Integer> slot_map_;
};

class VMShapeLowerMutator : public ExprMutator {
 public:
  static DataType ShapeDType() { return DataType::Int(64); }

  explicit VMShapeLowerMutator(IRModule mod) : ExprMutator(mod) {}

  IRModule Lower() {
    for (auto& p : builder_->GetContextIRModule()->functions) {
      Expr func = p.second;
      if (func->IsInstance<FunctionNode>()) {
        // prepare mapping and heap var
        expr2slot_ = PrimExprSlotCollector::Collect(Downcast<Function>(func));
        heap_size_ = IntImm(ShapeDType(), expr2slot_.size());
        shape_heap_ = Var("shape_heap", TensorStructInfo(ShapeExpr({heap_size_}), ShapeDType()));

        // mutate
        Function updated_func = Downcast<Function>(VisitExpr(func));
        builder_->UpdateFunction(p.first, updated_func);
      }
    }
    return builder_->GetContextIRModule();
  }

  void VisitBinding_(const MatchShapeNode* binding) override {
    Expr value = ExprMutator::VisitExpr(binding->value);

    // TODO(@yuchen): match_shape overloaded semantic: value is ShapeType
    Var shape = builder_->Emit(Call(ExternFunc("vm.builtin.shape_of"), {value}), "sh");
    StoreShape(shape, binding->pattern);
  }

  void VisitBinding_(const MatchCastNode* binding) override {
    // TODO(@tqchen): match_cast support for general struct info
    Expr value = ExprMutator::VisitExpr(binding->value);
    auto* tinfo = binding->struct_info.as<TensorStructInfoNode>();
    ICHECK(tinfo != nullptr) << "Match cast only support TensorStructInfo for now";
    auto* shape_expr = tinfo->shape.as<ShapeExprNode>();

    if (shape_expr) {
      bool dyn_shape = std::any_of(shape_expr->values.begin(), shape_expr->values.end(),
                                   [](const PrimExpr& e) { return !e->IsInstance<IntImmNode>(); });
      if (dyn_shape) {
        Var shape = builder_->Emit(Call(ExternFunc("vm.builtin.shape_of"), {value}), "sh");
        StoreShape(shape, shape_expr->values);
      }
    }
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const ShapeExprNode* node) override {
    if (IsConstantShape(GetRef<ShapeExpr>(node))) {
      return ExprMutator::VisitExpr_(node);
    }
    tir::PrimFunc func = CalculateShape(GetRef<ShapeExpr>(node));

    GlobalVar shape_func_var = builder_->AddFunction(func, "shape_func");
    builder_->Emit(Call(shape_func_var, {shape_heap_}), "_");

    // construct shape
    Array<Integer> indices;
    for (PrimExpr e : node->values) {
      indices.push_back(expr2slot_.at(e));
    }
    static const Op& load_shape_op = Op::Get("relax.vm.builtin.load_shape");
    auto load_shape_attr = make_object<ShapeHeapAttrs>();
    load_shape_attr->indices = indices;

    return builder_->Emit(Call(load_shape_op, {shape_heap_}, Attrs(load_shape_attr)), "sh");
  }

  Expr VisitExpr_(const FunctionNode* node) override {
    if (heap_size_->value > 0) {
      builder_->BeginBindingBlock();
      auto alloc_shape_heap = builder_->Normalize(
          Call(ExternFunc("vm.builtin.alloc_shape_heap"), {ShapeExpr({heap_size_})}));
      builder_->EmitNormalized(VarBinding(shape_heap_, alloc_shape_heap));
      for (Var param : node->params) {
        // TODO(relax-team): handle generalized case with tuple of Tensors
        if (auto* tensor_info = GetStructInfoAs<TensorStructInfoNode>(param)) {
          auto* shape_expr = tensor_info->shape.as<ShapeExprNode>();
          if (tensor_info->ndim != 0 && shape_expr) {
            Var shape = builder_->Emit(Call(ExternFunc("vm.builtin.shape_of"), {param}), "sh");
            StoreShape(shape, shape_expr->values);
          }
        }
      }
    }
    Expr new_body = this->VisitExpr(node->body);

    Array<BindingBlock> blocks;

    if (const SeqExprNode* seq = new_body.as<SeqExprNode>()) {
      if (heap_size_->value > 0) {
        blocks.push_back(builder_->EndBlock());
      }
      blocks.insert(blocks.end(), seq->blocks.begin(), seq->blocks.end());
      new_body = seq->body;
    }

    // FIXME(@yuchen): Implement vm.builtin.free_shape_heap.
    // builder_->Emit(Call(ExternFunc("vm.builtin.free_shape_heap"), {shape_heap_}), "gv");
    new_body = builder_->Normalize(SeqExpr(blocks, new_body));

    StructInfo ret_struct_info = node->ret_struct_info;

    // Because this pass is the last stage of build, ndim info is no longer needed for tensors.
    // The ret_type is weakened to unknown-dimensional DynTensorType.
    // TODO(@yuchen): change all tensor types in the function to unknown ndim
    if (const auto* tensor_sinfo = ret_struct_info.as<TensorStructInfoNode>()) {
      ret_struct_info = TensorStructInfo(tensor_sinfo->dtype, /*ndim=*/kUnknownNDim);
    }

    return builder_->Normalize(Function(node->params, new_body, ret_struct_info, node->attrs));
  }

  tir::PrimFunc CalculateShape(ShapeExpr s) {
    // TODO(ziheng): avoid generating shape func for known value
    tir::Var heap("heap", DataType::Handle());
    Array<PrimExpr> buffer_shape{heap_size_};
    tir::Buffer buffer = tir::decl_buffer(buffer_shape, ShapeDType(), "H");
    Map<tir::Var, tir::Buffer> buffer_map;
    buffer_map.Set(heap, buffer);

    Array<tir::Stmt> seq;
    for (PrimExpr e : s->values) {
      Map<tir::Var, PrimExpr> var_mapping = BuildVarMapping(e, buffer);
      PrimExpr value = tir::Substitute(e, var_mapping);
      // cast value to shape heap dtype
      if (value.dtype() != ShapeDType()) value = tir::Cast(ShapeDType(), value);
      Integer idx = expr2slot_.at(e);
      seq.push_back(tir::BufferStore(buffer, value, {idx}));
    }
    tir::Stmt body = tir::SeqStmt(seq);
    Array<tir::Var> params{heap};
    Type ret_type = VoidType();

    return tir::PrimFunc(params, body, ret_type, buffer_map);
  }

  Map<tir::Var, PrimExpr> BuildVarMapping(PrimExpr expr, tir::Buffer buffer) {
    Map<tir::Var, PrimExpr> ret;
    auto func = [&](const ObjectRef& e) {
      if (e->IsInstance<tir::VarNode>()) {
        PrimExpr prim_e = Downcast<PrimExpr>(e);
        tir::BufferLoad load(buffer, {expr2slot_.at(prim_e)});
        ret.Set(Downcast<tir::Var>(e), load);
      }
    };
    tir::PostOrderVisit(expr, func);
    return ret;
  }

  /*! \brief Store symbolic shape into indices of the VM shape heap. */
  void StoreShape(Expr shape, Array<PrimExpr> pattern) {
    static const Op& store_shape_op = Op::Get("relax.vm.builtin.store_shape");
    auto store_shape_attr = make_object<ShapeHeapAttrs>();

    Array<Integer> indices;
    for (size_t i = 0; i < pattern.size(); ++i) {
      auto it = expr2slot_.find(pattern[i]);
      ICHECK(it != expr2slot_.end()) << "PrimExpr pattern " << pattern[i] << " is not in expr2slot";
      indices.push_back((*it).second);
    }
    store_shape_attr->indices = indices;
    builder_->Emit(Call(store_shape_op, {shape, shape_heap_}, Attrs(store_shape_attr)), "gv");
  }

  bool IsConstantShape(ShapeExpr shape) const {
    for (PrimExpr e : shape->values) {
      if (!e->IsInstance<IntImmNode>()) {
        return false;
      }
    }
    return true;
  }

 private:
  // function-wise members
  IntImm heap_size_;
  Var shape_heap_;
  Map<PrimExpr, Integer> expr2slot_;
};

namespace transform {

Pass VMShapeLower() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return VMShapeLowerMutator(mod).Lower(); };
  return CreateModulePass(pass_func, 0, "VMShapeLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMShapeLower").set_body_typed(VMShapeLower);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
