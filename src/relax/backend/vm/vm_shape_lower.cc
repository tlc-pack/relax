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
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

class VMShapeLowerMutator : public ExprMutator {
 public:
  static DataType ShapeDType() { return DataType::Int(64); }

  explicit VMShapeLowerMutator(IRModule mod) : ExprMutator(mod) {}

  IRModule Lower() {
    for (auto& p : builder_->GetContextIRModule()->functions) {
      Expr func = p.second;
      if (func->IsInstance<FunctionNode>()) {
        // prepare mapping and heap var
        expr2slot_ = PrepareExpr2Slot(Downcast<Function>(func));
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
      builder_->Emit(VarBinding(
          shape_heap_, Call(ExternFunc("vm.builtin.alloc_shape_heap"), {ShapeExpr({heap_size_})})));

      for (Var param : node->params) {
        if (param->shape_.operator bool() && param->shape_.value().as<ShapeExprNode>()) {
          if (auto* param_type = param->checked_type_.as<DynTensorTypeNode>()) {
            if (param_type->ndim != 0) {
              Var shape = builder_->Emit(Call(ExternFunc("vm.builtin.shape_of"), {param}), "sh");
              StoreShape(shape, Downcast<ShapeExpr>(param->shape_.value())->values);
            }
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
      ret_struct_info = TensorStructInfo(tensor_sinfo->dtype, /*ndim=*/-1);
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

  Map<PrimExpr, Integer> PrepareExpr2Slot(Function expr) const {
    int cnt = 0;
    bool is_dyn_shape = false;
    Map<PrimExpr, Integer> ret;
    auto func = [&](const Expr& e) {
      if (e->IsInstance<ShapeExprNode>()) {
        ShapeExpr shape = Downcast<ShapeExpr>(e);
        for (auto prim_e : shape->values) {
          if (!prim_e->IsInstance<IntImmNode>()) {
            is_dyn_shape = true;
          }
          if (ret.count(prim_e) == 0) {
            ret.Set(prim_e, cnt++);
          }
        }
      }
    };
    PostOrderVisit(expr, func);

    // Avoid allocating shape heap and do shape computation for static-shape program
    if (!is_dyn_shape) {
      ret.clear();
    }
    return ret;
  }

  /*! \brief Store symbolic shape into indices of the VM shape heap. */
  void StoreShape(Expr shape, Array<PrimExpr> pattern) {
    static const Op& store_shape_op = Op::Get("relax.vm.builtin.store_shape");
    auto store_shape_attr = make_object<ShapeHeapAttrs>();

    Array<Integer> indices;
    for (size_t i = 0; i < pattern.size(); ++i) {
      Integer idx = expr2slot_.at(pattern[i]);
      indices.push_back(idx);
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
