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
 * \file src/relax/transform/shape_lower.cc
 * \brief 
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include "../../printer/text_printer.h"

namespace tvm {
namespace relax {

Array<ShapeExpr> CollectShapeExpr(Expr expr) {
  Array<ShapeExpr> ret;
  auto func = [&ret](const Expr& e) {
    if (e->IsInstance<ShapeExprNode>()) {
      ret.push_back(Downcast<ShapeExpr>(e));
    }
  };
  PostOrderVisit(expr, func);
  return ret;
} 

class ShapeReplacer : public ExprMutator {
 public:
  explicit ShapeReplacer(Map<ShapeExpr, Var> mapping) {
    mapping_ = mapping;
  }
  Expr VisitExpr_(const ShapeExprNode* op) override {
    return mapping_.at(GetRef<ShapeExpr>(op));
  }

 private:
  Map<ShapeExpr, Var> mapping_;
};

class ShapeLowerMutator : public ExprMutator {
 public:
  static DataType ShapeDType() {
    return DataType::Int(32);
  };

  explicit ShapeLowerMutator(IRModule mod) {
    mod_ = mod;
  }

  IRModule Lower() {
    ret_mod_ = IRModule();
    heap_counter_ = 0;
    shape_heap_ = Var("shape_heap", NullOpt, NullOpt);
    for (auto& p : mod_->functions) {
      GlobalVar var = p.first;
      Expr new_func = this->Mutate(p.second);
      ret_mod_->Add(var, Downcast<BaseFunc>(new_func));
    }
    return ret_mod_;
  } 

  void VisitMatchShape(const MatchShape& binding,
                       IRBuilder& builder) override {
    Expr value = binding->value;
    Array<PrimExpr> pattern = binding->pattern;
    Array<PrimExpr> indexes;
    for (size_t i = 0; i < pattern.size(); ++i) {
      IntImm idx(ShapeDType(), heap_counter_++);
      expr2idx_.Set(pattern[i], idx);
      indexes.push_back(idx);
    }
    ShapeExpr indexes_(indexes);
    Call call(ExternFunc("decode_shape"), {value, shape_heap_, indexes_});
    builder->Emit(call);
  }

  // Var VisitVarBinding(const VarBinding& binding,
  //                     IRBuilder& builder) override {
  //   Array<ShapeExpr> shapes = CollectShapeExpr(binding->value);
  //   return ExprMutator::VisitVarBinding(binding, builder);
  //   // map<ShapeExpr, Var> mapping;
  //   // for (shape : shapes) {
  //   //   vector<int> indexes = calculate_with_heap(shape);
  //   //   Var var = Emit(Call("construct_shape", indexes));
  //   //   mapping[shape] = var;
  //   // }
  //   // Replace(binding, shape, s);
  // }

  Expr VisitExpr_(const FunctionNode* node) override {
    Expr visited_func = ExprMutator::VisitExpr_(node);
    const auto* visited = visited_func.as<FunctionNode>();
    ICHECK(visited);
    const auto* seq = visited->body.as<SeqExprNode>();
    ICHECK(seq);

    // prologue block: allocate shape heap
    IntImm size(ShapeDType(), heap_size_);
    ShapeExpr heap_size({size});
    Call alloc_heap_call(ExternFunc("relax.alloc_shape_heap"), {heap_size});
    VarBinding binding(shape_heap_, alloc_heap_call);
    BindingBlock prologue({binding});


    // process body
    IRBuilder ib = IRBuilderNode::Create();
    Array<ShapeExpr> shapes = CollectShapeExpr(seq->body);
    Map<ShapeExpr, Var> mapping;
    for (ShapeExpr shape : shapes) {
      // call tir shape functions, result shape expr from heap
      tir::PrimFunc func = CalculateShape(shape);
      LOG(INFO) << "Generated TIR function: \n" <<  tir::AsTVMScript(func, false);
      // ib->Emit(Call(func, {shape_heap_}));
      // ret_mod_->Add(GlobalVar("shape_func" + std::to_string(shape_func_counter_++)), func);
      Array<PrimExpr> indexes;
      for (PrimExpr e : shape->values) {
        indexes.push_back(expr2idx_[e]);
      }
      ShapeExpr indexes_(indexes);
      Call call(ExternFunc("construct_shape"), {shape_heap_, indexes_});
      Var shape_var = ib->Emit(call);
      mapping.Set(shape, shape_var);
    }
    Expr new_body = ShapeReplacer(mapping).Mutate(seq->body);

    // epilogue block: kill the shape heap
    Call free_heap_call(ExternFunc("relax.free_shape_heap"), {shape_heap_});
    ib->Emit(free_heap_call);

    // process blocks
    Array<BindingBlock> blocks;
    blocks.push_back(prologue);
    blocks.insert(blocks.end(), seq->blocks.begin(), seq->blocks.end());
    blocks.push_back(ib->GetBlocks().back());


    SeqExpr new_seq(blocks, new_body);
    return Function(visited->name, visited->params, new_seq, visited->ret_type);
  }

  tir::PrimFunc CalculateShape(ShapeExpr s) {
    tir::Var heap("heap", DataType::Handle());
    Array<PrimExpr> buffer_shape{IntImm(DataType::Int(64), heap_size_)};
    tir::Buffer buffer = tir::decl_buffer(buffer_shape, ShapeDType(), "H");
    Map<tir::Var, tir::Buffer> buffer_map;
    buffer_map.Set(heap, buffer);

    Array<tir::Stmt> seq;
    for (PrimExpr e : s->values) {
      if (expr2idx_.count(e) == 0) {
        Map<tir::Var, PrimExpr> var_mapping = BuildVarMapping(e, buffer);
        PrimExpr value = tir::Substitute(e, var_mapping);
        IntImm idx(ShapeDType(), heap_counter_++);
        expr2idx_.Set(e, idx);
        seq.push_back(tir::Store(buffer->data, value, idx, tir::const_true()));
      }
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
        ICHECK(expr2idx_.count(prim_e) > 0);
        tir::Load load(ShapeDType(), buffer->data, expr2idx_.at(prim_e), tir::const_true());
        ret.Set(Downcast<tir::Var>(e), load);
      }
    };
    tir::PostOrderVisit(expr, func);
    return ret;
  } 

 private:
  IRModule mod_;
  IRModule ret_mod_;
  Var shape_heap_;
  int heap_size_{5};
  int heap_counter_;
  int shape_func_counter_{0};
  Map<PrimExpr, IntImm> expr2idx_;
  Array<Function> funcs_; 
};


TVM_REGISTER_GLOBAL("relax.transform.shape_lower")
.set_body_typed([](IRModule mod) {
  return ShapeLowerMutator(mod).Lower();
});

}  // namespace relax
}  // namespace tvm
