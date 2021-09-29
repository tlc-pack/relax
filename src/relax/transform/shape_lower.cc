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
  Expr Mutate(const Expr& expr) {
    heap_counter_ = 0;
    shape_heap_ = Var("shape_heap", NullOpt, NullOpt);
    return ExprMutator::Mutate(expr);
  }

  void VisitMatchShape(const MatchShape& binding,
                       IRBuilder& builder) override {
    Expr value = binding->value;
    Array<PrimExpr> pattern = binding->pattern;
    Array<PrimExpr> indexes;
    for (size_t i = 0; i < pattern.size(); ++i) {
      IntImm idx(DataType::Int(64), heap_counter_++);
      expr2idx_.Set(pattern[i], idx);
      indexes.push_back(idx);
    }
    ShapeExpr indexes_(indexes);
    Call call(ExternFunc("decode_shape"), {value, shape_heap_, indexes_});
    builder->Emit(call);
  }

  Var VisitVarBinding(const VarBinding& binding,
                      IRBuilder& builder) override {
    Array<ShapeExpr> shapes = CollectShapeExpr(binding->value);
    LOG(INFO) << "shapes: " << shapes;
    return ExprMutator::VisitVarBinding(binding, builder);
    // map<ShapeExpr, Var> mapping;
    // for (shape : shapes) {
    //   vector<int> indexes = calculate_with_heap(shape);
    //   Var var = Emit(Call("construct_shape", indexes));
    //   mapping[shape] = var;
    // }
    // Replace(binding, shape, s);
  }

  Expr VisitExpr_(const FunctionNode* node) override {
    Expr visited_func = ExprMutator::VisitExpr_(node);
    const auto* visited = visited_func.as<FunctionNode>();
    ICHECK(visited);
    const auto* seq = visited->body.as<SeqExprNode>();
    ICHECK(seq);
    LOG(INFO) << "SeqExpr body: " << seq->body;

    // prologue block: allocate shape heap
    IntImm size(DataType::Int(64), 5);
    ShapeExpr heap_size({size});
    Call alloc_heap_call(ExternFunc("relax.alloc_shape_heap"), {heap_size});
    VarBinding binding(shape_heap_, alloc_heap_call);
    BindingBlock prologue({binding});


    // process body
    IRBuilder ib = IRBuilderNode::Create();
    Array<ShapeExpr> shapes = CollectShapeExpr(seq->body);
    LOG(INFO) << "shapes: " << shapes;
    Map<ShapeExpr, Var> mapping;
    for (ShapeExpr shape : shapes) {
      // call tir shape functions, result shape expr from heap
      // PrimFunc func = CalculateShape(shape);
      // ib->Emit(Call(func, {shape_heap_}));
      // ib->Emit(Call(ExternFunc("shape_func"), {shape_heap_}));
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
    LOG(INFO) << "new body: " << new_body;

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

  // PrimFunc CalculateShape(ShapeExpr s) {
  //   tir::Var buffer("H", DataType::Handle());
  //   for (PrimExpr e : s) {
  //     if (expr2idx_.count(e) == 0) {
  //       // m -> H[0]
  //       Map<PrimExpr, PrimExpr> mapping = BuildVarMapping(e, buffer);
  //       PrimExpr value = Replace(e, var_mapping)
  //       IntImm idx(DataType::Int(64), heap_counter_++);
  //       expr2idx_.Set(e, idx);
  //       Stmt stm = Store(buffer, value, idx);
  //     }
  //   }
  //   Array<tir::Var> params;
  //   Stmt body;
  //   Type ret_type = VoidType();
  //   Map<tir::Var, Buffer> buffer_map;
  //   return PrimFunc(params, body, ret_type, buffer_map);
  // }

  // Map<PrimExpr, PrimExpr> BuildVarMapping(tir::PrimExpr e, tir::Var buffer) {
  //   Map<PrimExpr, PrimExpr> ret;
  //   auto func = [&ret](const PrimExpr& e) {
  //     if (e->IsInstance<tir::Var>()) {
  //       ICHECK(expr2idx_.count(e) > 0);
  //       ret.Set(e, Load(buffer, expr2idx_[e]));
  //     }
  //   };
  //   PostOrderVisit(expr, func);
  //   return ret;
  // } 

 private:
  Var shape_heap_;
  int heap_counter_;
  Map<PrimExpr, IntImm> expr2idx_;
  Array<Function> funcs_; 
};


TVM_REGISTER_GLOBAL("relax.transform.shape_lower")
.set_body_typed([](Expr expr) {
  return ShapeLowerMutator().Mutate(expr);
});

}  // namespace relax
}  // namespace tvm
