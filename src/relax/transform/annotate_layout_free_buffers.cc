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
#include <tvm/tir/stmt.h>

namespace tvm {
namespace relax {

/*! \brief collect the constants that are accessed only once.*/
class ConstantCollector : public ExprVisitor {
 public:
  static std::unordered_set<const ConstantNode*> GetOneAccessConstants(const IRModule& mod) {
    std::unordered_set<const ConstantNode*> ret;
    ConstantCollector collector;
    for (const auto& kv : mod->functions) {
      if (const auto* relax_f = kv.second.as<relax::FunctionNode>()) {
        collector(GetRef<Function>(relax_f));
      }
      for (const auto& kv : collector.access_counter_) {
        const ConstantNode* constant = kv.first;
        int counter = kv.second;
        if (counter == 1) {
          ret.insert(constant);
        }
      }
    }
    return ret;
  }

 private:
  void VisitExpr_(const ConstantNode* op) final {
    ExprVisitor::VisitExpr_(op);
    access_counter_[op] += 1;
  }
  /*! The counter for constants access. */
  std::unordered_map<const ConstantNode*, int> access_counter_;
};

class ConstantArgFinder : public ExprVisitor {
 public:
  static IRModule MarkLayoutRewriteAttr(const IRModule& mod) {
    auto one_access_constants = ConstantCollector::GetOneAccessConstants(mod);
    ConstantArgFinder finder(one_access_constants, mod);
    // Step 1. Visit all relax functions and detect candidates
    for (const auto& kv : mod->functions) {
      if (const auto* relax_f = kv.second.as<relax::FunctionNode>()) {
        finder(GetRef<Function>(relax_f));
      }
    }
    // Step 2. Update all PrimFunc candidates
    for (const auto& kv : mod->functions) {
      if (const auto* prim_func = kv.second.as<tir::PrimFuncNode>()) {
        auto it = finder.layout_free_attrs_.find(GetRef<tir::PrimFunc>(prim_func));
        if (it != finder.layout_free_attrs_.end()) {
          mod->Update(kv.first, WithAttr(GetRef<tir::PrimFunc>(prim_func),
                                         tir::attr::layout_free_buffers, (*it).second));
        }
      }
    }
    return mod;
  }

 private:
  ConstantArgFinder(const std::unordered_set<const ConstantNode*>& one_access_constants,
                    const IRModule& mod)
      : one_access_constants_(one_access_constants), mod_(mod) {}

  void VisitExpr_(const CallNode* op) final {
    ExprVisitor::VisitExpr_(op);
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    // Step 1. End function if it's not a call_tir
    if (op->op != call_tir_op_) {
      return;
    }

    // Step 2. Collect the argument indices of constants who are accessed once
    Array<Integer> layout_free_buffers;
    Array<Expr> call_tir_args = Downcast<Tuple>(op->args[1])->fields;
    for (size_t i = 0; i < call_tir_args.size(); i++) {
      if (const auto* constant = call_tir_args[i].as<ConstantNode>()) {
        if (one_access_constants_.count(constant)) {
          layout_free_buffers.push_back(i);
        }
      }
    }

    // Step 3. Find the corresponding PrimFunc
    GlobalVar gv = Downcast<GlobalVar>(op->args[0]);
    const Optional<tir::PrimFunc>& opt_f = MatchPrimFunc(gv);
    CHECK(opt_f.defined()) << "Cannot find the prim_func of the call_tir in the module: "
                           << op->args[0];
    const tir::PrimFunc& prim_func = opt_f.value();

    // Step 4. Add result to `layout_free_attrs_`
    auto it = layout_free_attrs_.find(prim_func);
    if (it == layout_free_attrs_.end()) {
      layout_free_attrs_.Set(prim_func, layout_free_buffers);
    } else {
      const Array<Integer>& cur_buffers = (*it).second;
      const Array<Integer>& intersection = GetIntersection(cur_buffers, layout_free_buffers);
      if (intersection.empty()) {
        layout_free_attrs_.erase(prim_func);
      } else {
        layout_free_attrs_.Set(prim_func, intersection);
      }
    }
  }

 private:
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

  static Array<Integer> GetIntersection(const Array<Integer>& lhs, const Array<Integer>& rhs) {
    Array<Integer> result;
    for (const Integer& x : lhs) {
      if (std::any_of(rhs.begin(), rhs.end(),
                      [&x](const Integer& y) -> bool { return x->value == y->value; })) {
        result.push_back(x);
      }
    }
    return result;
  }

  std::unordered_set<const ConstantNode*> one_access_constants_;
  Map<tir::PrimFunc, Array<Integer>> layout_free_attrs_;
  const IRModule& mod_;
};

namespace transform {
Pass AnnotateLayoutFreeBuffers() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return ConstantArgFinder::MarkLayoutRewriteAttr(m); };
  return CreateModulePass(/*pass_function=*/pass_func,                //
                          /*opt_level=*/0,                            //
                          /*pass_name=*/"AnnotateLayoutFreeBuffers",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.AnnotateLayoutFreeBuffers")
    .set_body_typed(AnnotateLayoutFreeBuffers);

}  // namespace transform
}  // namespace relax
}  // namespace tvm