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

#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include <utility>

#include "../analysis/liveness_analysis.h"

namespace tvm {
namespace relax {

class KillInserter : public ExprMutator {
 public:
  KillInserter(const ControlFlowGraph* cfg, const LivenessAnalysis* lva, Array<Var> params)
      : cfg_(cfg), lva_(lva), params_(std::move(params)) {}

  // Limitations
  // -----------
  // (1) For simplicity, we only insert kills when visiting binding nodes, and always emit the
  // kill as a single subsequent binding. In the case of an `If` expression then the result is
  // in the body of a SeqExpr. Therefore, there is no binding after which to emit the
  // final kill in the `If/Else` block:
  // if cond:
  //    tensor_3 = relax.builtin.alloc_tensor(...)
  //    output = relax.add(s, tensor_3)
  //    ^^ Result of true block is relax.add(..) and there is no place to
  //       insert kill_tensor(tensor_3)
  // else:
  //    tensor_4 = relax.builtin.alloc_tensor(...)
  //    intermediate_result = relax.add(s, tensor_4)
  //    kill_tensor: Tuple() = relax.memory.kill_tensor(tensor_4)
  //    output = intermediate_result
  //    ^^ As opposed to this case where the body of the SeqExpr is a Var.
  //
  // (2) Killed variables are calculated as live in - live out, which misses variables that are
  // actually dead but not in a live-in set. Example:
  //  @R.function
  //  def func(i: Tensor((), "float32"), s: Tensor((2, 3), "float32")) ->
  //      Tensor(None, "float32", ndim = 2):
  //    tensor_1 = relax.builtin.alloc_tensor(...)
  //    unused_var: Tensor((), "float32") = relax.add(i, tensor_1)
  //    ^^ No kill is inserted for unused_var because it is not in a live-in set.
  //    output: Tensor((2, 3), "float32") = relax.add(s, tensor_1)
  //    kill_tensor: Tuple() = relax.memory.kill_tensor(tensor_1)
  //    return output
  //
  // (3) Variable aliasing (including done through match_shape) and tuples are not yet supported.
  // -----------

  void VisitBinding_(const VarBindingNode* binding_node) override {
    auto binding = runtime::GetRef<VarBinding>(binding_node);
    // TODO(gigiblender): Handle aliasing.
    ICHECK(!binding_node->value.as<VarNode>()) << "aliasing is not supported.";
    ICHECK(cfg_->var_binding_map.count(binding)) << "all Binding exprs should be mapped in the CFG";

    const ControlFlowGraph::NodePtr n = cfg_->var_binding_map.at(binding);
    const VarSet& li = lva_->live_in.at(n);
    const VarSet& lo = lva_->live_out.at(n);

    // Killed vars = live in - live out.
    VarSet kills;
    for (const Var& v : li) {
      if (!lo.count(v)) {
        kills.insert(v);
      }
    }

    ExprMutator::VisitBinding_(binding_node);
    emitKills(kills);
  }

  void VisitBinding_(const MatchShapeNode* shape_node) override {
    auto binding = runtime::GetRef<MatchShape>(shape_node);
    // TODO(gigiblender): Handle aliasing.
    ICHECK(!shape_node->value.as<VarNode>()) << "aliasing is not supported.";
    ICHECK(cfg_->var_binding_map.count(binding)) << "all Binding exprs should be mapped in the CFG";

    const ControlFlowGraph::NodePtr n = cfg_->var_binding_map.at(binding);

    const VarSet& li = lva_->live_in.at(n);
    const VarSet& lo = lva_->live_out.at(n);

    // Killed vars = live in - live out.
    VarSet kills;
    for (const Var& v : li) {
      if (!lo.count(v)) {
        kills.insert(v);
      }
    }

    ExprMutator::VisitBinding_(shape_node);
    emitKills(kills);
  }

 private:
  void emitKills(const VarSet& kills) {
    static const Op& memory_kill_tensor_op = Op::Get("relax.memory.kill_tensor");

    for (const Var& v : kills) {
      // Do not emit a kill if this is a parameter
      if (std::count(params_.begin(), params_.end(), v)) {
        continue;
      }
      // Do not emit a kill for the var if it is bound to the return of a call packed.
      if (cfg_->call_packed_returns.count(v)) {
        continue;
      }
      builder_->Emit(Call(memory_kill_tensor_op, {v}, {}), "kill_tensor");
    }
  }

  const ControlFlowGraph* cfg_;
  const LivenessAnalysis* lva_;
  Array<Var> params_;
};

namespace transform {

Pass InsertKills() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        Arena arena;
        ControlFlowGraph cfg = ControlFlowGraph::Create(&arena, f);
        UseDefAnalysis use_def = UseDefAnalysis::Analyze(cfg);
        LivenessAnalysis lva = LivenessAnalysis::Analyze(cfg, use_def);
        KillInserter ki(&cfg, &lva, f->params);
        auto nf = Downcast<Function>(ki.VisitExpr(f));
        return nf;
      };
  return CreateFunctionPass(pass_func, 0, "NaivePlanMemory", {});
}

TVM_REGISTER_GLOBAL("relax.transform.InsertMemoryKills").set_body_typed(InsertKills);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
