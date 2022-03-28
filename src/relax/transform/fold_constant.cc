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

#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

class ConstantFolder : public ExprMutator {
 public:
  ConstantFolder(IRModule ctx_module) : ctx_module_(ctx_module) {}

 private:
  /*!
   * \brief Pattern match expr to a constant shape and get runtime shape tuple from it.
   * \return The runtime shape tuple, or nullopt if it is not a constant shape.
   */
  static Optional<runtime::ShapeTuple> MatchConstShape(const Expr& expr) {
    if (auto* shape = expr.as<ShapeExprNode>()) {
      std::vector<int64_t> shape_values;
      for (const auto v : shape->values) {
        if (auto* ptr = v.as<IntImmNode>()) {
          shape_values.push_back(ptr->value);
        } else {
          return NullOpt;
        }
      }
      return runtime::ShapeTuple(shape_values.begin(), shape_values.end());
    } else {
      return NullOpt;
    }
  }

  /*!
   * \brief Pattern match op to constant array arguments.
   * \return The constant array arguments, or nullopt if match fails.o
   */
  static Optional<Array<runtime::NDArray>> MatchConstArrayArgs(const Array<Expr>& args) {
    Array<runtime::NDArray> res;
    for (auto arg : args) {
      if (auto* ptr = arg.as<relay::ConstantNode>()) {
        res.push_back(ptr->data);
      } else {
        return NullOpt;
      }
    }
    return res;
  }

  /*!
   * \brief Pattern match op to a TIR function and look it up.
   * \return The TIR function, or nullopt if patter match fails.
   */
  Optional<tir::PrimFunc> MatchPrimFunc(const Expr& op) {
    if (auto* ptr = op.as<GlobalVarNode>()) {
      // NOTE: as check works for nullptr(returns null)
      Optional<BaseFunc> base_func = ctx_module_->functions.Get(GetRef<GlobalVar>(ptr));
      if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
        return GetRef<tir::PrimFunc>(pfunc);
      }
    }
    return NullOpt;
  }

  /*!
   * \brief Get a cached build version of func
   * \return The cached func, nullopt if func cannot be built.
   */
  Optional<PackedFunc> GetCachedBuild(tir::PrimFunc func) {
    // TODO(tvm-team): consider another way of bulk extract and build PrimFunc once
    // would be helpful for future cases where PrimFunc recursively call into each other
    Target eval_cpu_target{"llvm"};

    auto it = func_build_cache_.find(func);
    if (it != func_build_cache_.end()) {
      return it->second;
    }
    Optional<PackedFunc> build_func = NullOpt;

    try {
      runtime::Module rt_module =
          build(LowerPrimFunc(func, "tir_function"), eval_cpu_target, eval_cpu_target);
      build_func = rt_module.GetFunction("tir_function");
    } catch (const tvm::Error& err) {
      // build failure may happen in which case we skip
      DLOG(WARNING) << "Build failure for function " << func;
    }
    func_build_cache_[func] = build_func;
    return build_func;
  }

  // Try constant evaluate the function call
  // if failed return NullOpt
  Optional<Expr> ConstEvaluateCallTIR(tir::PrimFunc tir_func, Array<runtime::NDArray> arr_args,
                                      runtime::ShapeTuple shape) {
    // obtain function from the cache.
    Optional<PackedFunc> func = GetCachedBuild(tir_func);
    if (!func) return NullOpt;

    std::vector<TVMValue> values(arr_args.size() + 1);
    std::vector<int> type_codes(arr_args.size() + 1);

    DLDevice cpu_dev = {DLDeviceType::kDLCPU, 0};
    runtime::NDArray ret_tensor = runtime::NDArray::Empty(shape, DataType::Float(32), cpu_dev);

    // avoid set rvalue ref which get de-allocated later, store args in a vector
    // where temp_args[i] are lvalue ref that is stable
    std::vector<runtime::NDArray> temp_args(arr_args.begin(), arr_args.end());

    size_t arg_offset = 0;
    for (; arg_offset < arr_args.size(); ++arg_offset) {
      runtime::TVMArgsSetter(values.data(), type_codes.data())(arg_offset, temp_args[arg_offset]);
    }
    // set return value
    runtime::TVMArgsSetter(values.data(), type_codes.data())(arg_offset++, ret_tensor);

    TVMRetValue ret;
    // invoke
    func.value().CallPacked(TVMArgs(values.data(), type_codes.data(), values.size()), &ret);
    return Constant(ret_tensor);
  }

  Expr VisitCallTIR(Call call) {
    // call_tir needs to have at least three arguments
    ICHECK_GE(call->args.size(), 3);
    Optional<tir::PrimFunc> func = MatchPrimFunc(call->args[0]);
    ICHECK(call->args[1].as<TupleNode>()) << "call_tir.args[1] requires to be Tuple";
    Optional<Array<runtime::NDArray>> arr_args =
        MatchConstArrayArgs(call->args[1].as<TupleNode>()->fields);
    Optional<runtime::ShapeTuple> shape = MatchConstShape(call->args[2]);

    // Pattern 0: call constant function, const argument with const shape.
    if (func && arr_args && shape) {
      // value_or will return value if it is not null, otherwise return or
      return ConstEvaluateCallTIR(func.value(), arr_args.value(), shape.value()).value_or(call);
    }
    // TODO(hongyi): support const-fold tuple outputs
    return std::move(call);
  }

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Call post_call = Downcast<Call>(VisitExprPostOrder_(call));
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    if (call->op.same_as(call_tir_op)) {
      return VisitCallTIR(post_call);
    } else {
      return std::move(post_call);
    }
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    Optional<Expr> opt = LookupBinding(GetRef<Var>(op));
    // NOTE: opt can be nullptr, in which case opt is nullptr
    // as check checks if opt is not null and is instance of constant
    if (opt.as<relay::ConstantNode>()) {
      return opt.value();
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    Optional<Expr> opt = LookupBinding(GetRef<Var>(op));
    // NOTE: opt can be nullptr, in which case opt is nullptr
    // as check checks if opt is not null and is instance of constant
    if (opt.as<relay::ConstantNode>()) {
      return opt.value();
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  // the context module to lookup functions
  IRModule ctx_module_;
  // cache for function build, via structural equality
  std::unordered_map<tir::PrimFunc, Optional<runtime::PackedFunc>, StructuralHash, StructuralEqual>
      func_build_cache_;
};

namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        ConstantFolder folder(m);
        return Downcast<Function>(folder(f));
      };
  return CreateFunctionPass(pass_func, 0, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
