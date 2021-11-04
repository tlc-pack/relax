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
 * \brief
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/relax/attrs/shape.h>

namespace tvm {
namespace relax {
namespace vm {

class VMShapeLowerMutator : public ExprMutator {
 public:
  static DataType ShapeDType() {
    return DataType::Int(64);
  };

  explicit VMShapeLowerMutator(IRModule mod) { mod_ = mod; }

  IRModule Lower() {
    ret_mod_ = IRModule();
    for (auto& p : mod_->functions) {
      Expr func = p.second;
      if (p.second->IsInstance<FunctionNode>()) {
        // prepare mapping and heap var
        expr2slot_ = PrepareExpr2Slot(Downcast<Function>(func));
        heap_size_ = IntImm(ShapeDType(), expr2slot_.size());
        DynTensorType heap_type(1, ShapeDType());
        shape_heap_ = Var("shape_heap", ShapeExpr({heap_size_}), heap_type);

        // mutate
        func = this->Mutate(func);
      }
      ret_mod_->Add(p.first, Downcast<BaseFunc>(func));
    }
    return ret_mod_;
  }

  void VisitMatchShape(const MatchShape& binding) override {
    Expr shape = ExprMutator::VisitExpr(binding->value);
    static const Op& store_shape_op = Op::Get("relax.vm.builtin.store_shape");
    auto store_shape_attr = make_object<ShapeHeapAttrs>();

    Array<PrimExpr> pattern = binding->pattern;
    Array<Integer> indices;
    for (size_t i = 0; i < pattern.size(); ++i) {
      int idx = expr2slot_.at(pattern[i]);
      indices.push_back(idx);
    }
    store_shape_attr->indices = indices;
    builder_->Emit(Call(store_shape_op, {shape, shape_heap_}, Attrs(store_shape_attr)), "gv");
  }

  Expr VisitExpr_(const ShapeExprNode* node) override {
    if (IsConstantShape(GetRef<ShapeExpr>(node))) {
      return ExprMutator::VisitExpr_(node);
    }
    tir::PrimFunc func = CalculateShape(GetRef<ShapeExpr>(node));
    std::string shape_func_name = name_table_->GetUniqueName("shape_func");
    func = WithAttr(std::move(func), "global_symbol", runtime::String(shape_func_name));
    GlobalVar shape_func_var(shape_func_name);
    // TODO make sure shape_heap doesnt get redefined by local funcs?
    builder_->Emit(Call(shape_func_var, {shape_heap_}), "_compute_shape");
    ret_mod_->Add(shape_func_var, func);

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
    Array<Var> params;
    for (Var param : node->params) {
      params.push_back(Downcast<Var>(this->Mutate(param)));
    }
    Type ret_type = this->VisitType(node->ret_type);

    builder_->BeginBindingBlock();
    builder_->Emit(VarBinding(
        shape_heap_, Call(ExternFunc("vm.builtin.alloc_shape_heap"), {ShapeExpr({heap_size_})})));

    Expr new_body = this->Mutate(node->body);

    Array<BindingBlock> blocks;

    if (const SeqExprNode* seq = new_body.as<SeqExprNode>()) {
      blocks.push_back(builder_->EndBlock());
      blocks.insert(blocks.end(), seq->blocks.begin(), seq->blocks.end());
      builder_->BeginBindingBlock();
      new_body = seq->body;
    }

    // FIXME(@yuchen): Implement vm.builtin.free_shape_heap.
    // builder_->Emit(Call(ExternFunc("vm.builtin.free_shape_heap"), {shape_heap_}), "gv");
    blocks.push_back(builder_->EndBlock());
    new_body = SeqExpr(blocks, new_body);

    return Function(node->name, params, new_body, ret_type);
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
      int idx = expr2slot_.at(e);
      seq.push_back(tir::Store(buffer->data, value, idx, tir::const_true()));
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
        tir::Load load(ShapeDType(), buffer->data, expr2slot_.at(prim_e), tir::const_true());
        ret.Set(Downcast<tir::Var>(e), load);
      }
    };
    tir::PostOrderVisit(expr, func);
    return ret;
  }

  Map<PrimExpr, Integer> PrepareExpr2Slot(Function expr) const {
    int cnt = 0;
    Map<PrimExpr, Integer> ret;
    auto func = [&](const Expr& e) {
      if (e->IsInstance<ShapeExprNode>()) {
        ShapeExpr shape = Downcast<ShapeExpr>(e);
        for (auto prim_e : shape->values) {
          if (ret.count(prim_e) == 0) {
            ret.Set(prim_e, cnt++);
          }
        }
      }
    };
    PostOrderVisit(expr, func);
    return ret;
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
  IRModule mod_;
  IRModule ret_mod_;
  int shape_func_counter_{0};

  // function-wise members
  IntImm heap_size_;
  Var shape_heap_;
  Map<PrimExpr, Integer> expr2slot_;
};

TVM_REGISTER_GLOBAL("relax.transform.vm_shape_lower")
.set_body_typed([](IRModule mod) {
  return VMShapeLowerMutator(mod).Lower();
});

}  // namespace vm
}  // namespace relax
}  // namespace tvm
