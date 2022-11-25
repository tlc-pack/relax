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
 * \file src/relax/transform/fma_rewrite.cc
 * \brief Perform fused multiply-add rewriting.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class VtxMMRewriter : public ExprMutator {
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* _call) override {
    static const Op& op_vtx_mm = Op::Get("relax.vtx_mm");
    static const Op& call_tir = Op::Get("relax.call_tir");

    Call call = Downcast<Call>(VisitExprPostOrder_(_call));
    if (!call->op.same_as(op_vtx_mm)) {
      return call;
    }
    const auto* attr = call->attrs.as<VtxMMAttrs>();
    const auto* f = runtime::Registry::Get("tvm.relax.vtx.cutlass_gemm");
    ICHECK(f != nullptr);
    ICHECK(attr != nullptr);
    ICHECK(call->args.size() == 2 || call->args.size() == 3);
    bool has_bias = call->args.size() == 3;
    std::string func_name = "vtx_mm_" + std::to_string(counter_++);
    LOG(INFO) << "====== " << func_name << " ======";
    LOG(INFO) << "a->shape = " << Downcast<ShapeExpr>(call->args[0]->shape());
    LOG(INFO) << "b->shape = " << Downcast<ShapeExpr>(call->args[1]->shape());
    if (has_bias) {
      LOG(INFO) << "bias->shape = " << Downcast<ShapeExpr>(call->args[2]->shape());
    }
    // HACK: assume [1, m, k] * [1, n, k] => [1, m, n]
    int m = Downcast<IntImm>(Downcast<ShapeExpr>(call->args[0]->shape())->values[1])->value;
    int k = Downcast<IntImm>(Downcast<ShapeExpr>(call->args[0]->shape())->values[2])->value;
    int n = Downcast<IntImm>(Downcast<ShapeExpr>(call->args[1]->shape())->values[1])->value;
    // HACK: assume f32, f32 -> f32
    std::string type_a = "float32";
    std::string type_b = "float32";
    std::string type_c = "float32";
    // HACK: assume row, col -> row
    std::string layout_a = "row";
    std::string layout_b = "col";
    std::string layout_c = "row";
    // HACK: assume it's just dense
    std::string op_type = "cutlass.dense";
    if (!attr->epilogue_pattern.empty()) {
      op_type = attr->epilogue_pattern;
    }
    LOG(INFO) << "op_type = " << op_type;
    LOG(INFO) << "out = " << call->shape();
    // Call cutlass tuner
    std::string source = (*f)(func_name,                     //
                              m, n, k,                       //
                              type_a, type_b, type_c,        //
                              layout_a, layout_b, layout_c,  //
                              op_type, has_bias);

    const static constexpr char* kCSource = "c_source";
    const static constexpr char* kCSourceFmt = "c_source_fmt";
    const static constexpr char* kCSourceFmtCuda = "cu";
    ExternFunc func(func_name);
    func = WithAttrs(std::move(func),  //
                     Map<String, ObjectRef>{
                         {String(kCSource), String(source)},
                         {String(kCSourceFmt), String(kCSourceFmtCuda)},
                     });
    GlobalVar gv = this->builder_->AddFunction(func, func_name);
    new_funcs.Set(gv, func);  // HACK: somehow the line above doesn't work with FunctionPass
    return Call(call_tir, {gv, Tuple(call->args), ShapeExpr({1, m, n})}, Attrs(),
                {
                    DynTensorType(3, DataType::Float(32)),
                });
  }

  int counter_ = 0;

 public:
  Map<GlobalVar, BaseFunc> new_funcs;
};

namespace transform {

Pass LowerVtxMM() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) -> IRModule {
    VtxMMRewriter rewriter;
    Map<GlobalVar, BaseFunc> funcs;
    for (auto kv : m->functions) {
      GlobalVar gv = kv.first;
      BaseFunc func = kv.second;
      if (const auto* relax_func = func.as<FunctionNode>()) {
        ObjectPtr<FunctionNode> new_func = make_object<FunctionNode>(*relax_func);
        new_func->body = rewriter.VisitExpr(relax_func->body);
        funcs.Set(gv, Function(new_func));
      } else {
        funcs.Set(gv, func);
      }
    }
    for (auto kv : rewriter.new_funcs) {
      funcs.Set(kv.first, kv.second);
    }
    return IRModule(funcs);
  };
  return tvm::transform::CreateModulePass(pass_func, 2, "LowerVtxMM", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LowerVtxMM").set_body_typed(LowerVtxMM);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
