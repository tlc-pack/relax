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
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

/*! \brief Collect Layout Rewrite Preproc blocks and create separate functions. */
class PreprocFunctionCreator : public StmtExprVisitor {
 public:
  /*!
   * \brief Create PrimFuncs for layout rewrite preproc.
   * \param func The original input function.
   * \return A map from weight buffer to it's rewrite funcs.
   */
  static Map<Buffer, PrimFunc> GetFuncs(const PrimFunc& func) {
    PreprocFunctionCreator creator;
    CHECK(func->body->IsInstance<BlockRealizeNode>())
        << "Schedulable PrimFuncs are expected. i.e.the body of function should be a root block.";
    BlockRealize root_realize = Downcast<BlockRealize>(func->body);
    creator(root_realize->block->body);
    return creator.preproc_funcs_;
  }

 private:
  explicit PreprocFunctionCreator() = default;

  void CreatePreprocFunction(const BlockRealizeNode* op) {
    const Block& block = op->block;
    CHECK(block->reads.size() == 1 && block->writes.size() == 1)
        << "A layout rewrite block is expect to have only one input and one output.";

    // Step 1. Create a new block and remove "preproc" attrs
    auto block_ptr = make_object<BlockNode>(*block.get());
    block_ptr->annotations.erase(tir::attr::meta_schedule_layout_rewrite_preproc);
    Block new_block(block_ptr);

    // Step 2. Create block realize and loop nesting
    auto realize_ptr = make_object<BlockRealizeNode>(*op);
    realize_ptr->block = new_block;
    Stmt body = BlockRealize(realize_ptr);

    for (int i = static_cast<int>(loop_stack_.size()) - 1; i >= 0; i--) {
      auto for_ptr = make_object<ForNode>(*(loop_stack_[i].get()));
      for_ptr->body = body;
      body = For(for_ptr);
    }

    // Step 3. Create new functions
    // Step 3.1. Function body
    body = Block({}, {}, {}, "root", body);
    body = BlockRealize({}, Bool(true), Downcast<Block>(body));
    // Step 3.2. Function Params
    tir::Var src_arg("src", PrimType(DataType::Handle()));
    tir::Var tgt_arg("tgt", PrimType(DataType::Handle()));
    // Step 3.3. Function buffer_map
    const Buffer& src_buffer = new_block->reads[0]->buffer;
    const Buffer& tgt_buffer = new_block->writes[0]->buffer;
    Map<tir::Var, Buffer> buffer_map{{src_arg, src_buffer}, {tgt_arg, tgt_buffer}};
    PrimFunc func(/*params=*/{src_arg, tgt_arg},
                  /*body=*/std::move(body),
                  /*ret_type=*/VoidType(),
                  /*buffer_map=*/std::move(buffer_map));
    func = tir::RenewDefs(func);

    // Step 4. Store the result to the output map
    // Note: we use original buffer as key rather than the new created buffer.
    preproc_funcs_.Set(block->reads[0]->buffer, std::move(func));
  }

  void VisitStmt_(const ForNode* op) final {
    loop_stack_.push_back(GetRef<For>(op));
    StmtVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    // preproc block will only appear directly under root block, so we don't recursively visit
    Block block = op->block;
    auto it = block->annotations.find(tir::attr::meta_schedule_layout_rewrite_preproc);
    if (it != block->annotations.end() && is_one(Downcast<PrimExpr>((*it).second))) {
      CreatePreprocFunction(op);
    }
  }

  /*! \brief loop stack */
  Array<For> loop_stack_;
  /*! \brief The created preproc functions. */
  Map<Buffer, PrimFunc> preproc_funcs_;
};
}  // namespace tir

namespace relax {

class SplitPreprocMutator : public ExprMutator {
 public:
  static IRModule Transform(IRModule mod) {
    SplitPreprocMutator mutator(mod);
    // Step 1. Visit relax functions and insert separate layout-rewrite steps
    for (const auto& kv : mod->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& func = kv.second;
      if (kv.second.as<relax::FunctionNode>()) {
        auto updated_func = Downcast<Function>(mutator(func));
        mutator.builder_->UpdateFunction(gv, updated_func);
      }
    }
    mod = mutator.builder_->GetContextIRModule();
    // Step 2. Remove weight layout rewrite block inside original PrimFuncs
    mod = tir::transform::RemoveWeightLayoutRewriteBlock()(mod);
    return mod;
  }

 private:
  explicit SplitPreprocMutator(const IRModule& mod) : ExprMutator(mod), mod_(mod) {}

  /*!
   * \brief Pattern match op to a TIR function and look it up.
   * \return The TIR function, or NullOpt if patter match fails.
   */
  Optional<tir::PrimFunc> MatchPrimFunc(const GlobalVar& gv) const {
    Optional<BaseFunc> base_func = mod_->functions.Get(gv);
    if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
      return GetRef<tir::PrimFunc>(pfunc);
    } else {
      return NullOpt;
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op == call_tir_op) {
      return VisitCallTIR(call);
    } else {
      return GetRef<Expr>(call);
    }
  }

  Expr VisitCallTIR(const CallNode* call) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    ICHECK(call->op == call_tir_op);

    // Step 1. Get PrimFunc and Create preproc function if possible.
    Optional<tir::PrimFunc> opt_f = MatchPrimFunc(Downcast<GlobalVar>(call->args[0]));
    CHECK(opt_f.defined()) << "Cannot find PrimFuncs used in call_tir";
    const tir::PrimFunc& f = opt_f.value();
    Map<tir::Buffer, tir::PrimFunc> preproc_funcs = tir::PreprocFunctionCreator::GetFuncs(f);

    // Step 2. Updating all layout free buffers
    Array<Expr> call_tir_args = GetTIRArgs(call);
    for (const auto& kv : preproc_funcs) {
      const tir::Buffer& buffer = kv.first;
      const tir::PrimFunc& func = kv.second;
      // Step 2.1 Get the index of the layout free buffer
      int idx = GetBufferIndex(f, buffer);
      ICHECK_LE(idx, call_tir_args.size());
      // The layout rewrite function has only one input and one output.
      ICHECK_EQ(func->params.size(), 2);
      // Step 2.2 Emit layout rewrite preproc function to relax
      const ShapeExpr& out_shape = GetOutputShapeFromPreprocFunc(func);
      const GlobalVar& layout_rewrite_func = builder_->AddFunction(func, "layout_rewrite");
      const Type& type = DynTensorType(out_shape->values.size(), buffer->dtype);
      const Array<Expr> args = {layout_rewrite_func, Tuple({call_tir_args[idx]}), out_shape};
      Var new_var = builder_->Emit(Call(/*op=*/call_tir_op,
                                        /*args=*/args,
                                        /*attrs=*/{},
                                        /*type_args=*/{type}));
      // Step 2.3 update call_tir arguments to use rewritten weight
      call_tir_args.Set(idx, new_var);
    }
    // Step 3. Updating original call and use rewritten weight expr.
    return Call(/*op=*/call_tir_op,                                             //
                /*args=*/{call->args[0], Tuple(call_tir_args), call->args[2]},  //
                /*attrs=*/{},                                                   //
                /*type_args=*/call->type_args);
  }

  static ShapeExpr GetOutputShapeFromPreprocFunc(const tir::PrimFunc& func) {
    ICHECK_EQ(func->params.size(), 2);
    tir::Buffer buffer = func->buffer_map.Get(func->params[1]).value();
    // Convert to i64 if is constant
    // TODO(Siyuan): support symbolic shape
    Array<PrimExpr> shape;
    for (const PrimExpr& e : buffer->shape) {
      if (const auto* imm = e.as<IntImmNode>()) {
        shape.push_back(tir::make_const(DataType::Int(64), imm->value));
      } else {
        shape.push_back(e);
      }
    }
    return ShapeExpr(shape);
  }

  static Array<Expr> GetTIRArgs(const CallNode* call) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    ICHECK_EQ(call->op, call_tir_op);
    Array<Expr> ret;
    if (call->args[1].as<TupleNode>()) {
      ret = Downcast<Tuple>(call->args[1])->fields;
    } else {
      ret = {call->args[1]};
    }
    return ret;
  }

  static int GetBufferIndex(const tir::PrimFunc& f, const tir::Buffer& buffer) {
    for (size_t i = 0; i < f->params.size(); ++i) {
      auto it = f->buffer_map.find(f->params[i]);
      if (it != f->buffer_map.end() && (*it).second.same_as(buffer)) {
        return i;
      }
    }
    LOG(FATAL) << "Can not find buffer " << buffer << " in the given PrimFunc";
    return -1;
  }

  const IRModule& mod_;
};
namespace transform {
Pass SplitLayoutRewritePreproc() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return SplitPreprocMutator::Transform(m); };
  return CreateModulePass(/*pass_function=*/pass_func,                //
                          /*opt_level=*/0,                            //
                          /*pass_name=*/"SplitLayoutRewritePreproc",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.SplitLayoutRewritePreproc")
    .set_body_typed(SplitLayoutRewritePreproc);
}  // namespace transform
}  // namespace relax
}  // namespace tvm