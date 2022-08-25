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
 * \file tvm/relax/transform/meta_schedule.cc
 * \brief Pass for meta_schedule tuning
 */
#include <tvm/relax/transform.h>
#include <tvm/relax/tuning_api.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace relax {
class MetaScheduleTuner {
 public:
  explicit MetaScheduleTuner(Target target, Array<ObjectRef> config, String work_dir,
                             const tvm::meta_schedule::Database& database)
      : target_(target), config_(config), work_dir_(work_dir), database_(database) {
    candgen_func_ = runtime::Registry::Get("relax.tuning_api.default_generate_candidate");
    ICHECK(candgen_func_) << "Default candidate generation function is not found.";
  }

  // TODO(@sunggg): Currently, only supports basic arguments.
  IRModule TuneIRMod(IRModule mod, transform::PassContext ctx) {
    Trace trace = Downcast<Trace>(ctx->GetCurrentTrace());
    ctx->PopTrace();
    Choice choice("meta_schedule.tune_relax_irmod_with_tuning_api",
                  {target_, config_, work_dir_, database_},
                  "relax.tuning_api.Choice.default_constr_func", {});
    Knob knob("meta_schedule.tune_irmod", {{"0", choice}});
    Array<Trace> candidates = (*candgen_func_)(Array<Knob>({knob}), trace);
    ICHECK(candidates.size() == 1);
    Trace best_trace = candidates[0];
    ctx->PushTrace(best_trace);
    return best_trace->out_mod;
  }

  // TODO(@sunggg): Currently, only supports basic arguments.
  tir::PrimFunc TuneTIR(tir::PrimFunc f, transform::PassContext ctx) {
    auto parse_mod_func = runtime::Registry::Get("tvm.meta_schedule.tune.parse_mod");
    ICHECK(parse_mod_func) << "Parse function is not found.";
    // TODO(@sunggg): Whenever we tune tir, assume we start a new trace w/o pushing to the trace
    // stack. Revisit later when we collect more usecases.
    Trace trace = Trace((*parse_mod_func)(f), {}, {});

    Choice choice("meta_schedule.tune_tir_with_tuning_api",
                  {target_, config_, work_dir_, database_},
                  "relax.tuning_api.Choice.default_constr_func", {});
    Knob knob("meta_schedule.tune_irmod", {{"0", choice}});
    Array<Trace> candidates = (*candgen_func_)(Array<Knob>({knob}), trace);
    ICHECK(candidates.size() == 1);
    Trace best_trace = candidates[0];
    auto gvars = best_trace->out_mod->GetGlobalVars();
    ICHECK(gvars.size() == 1);
    tvm::BaseFunc new_func = best_trace->out_mod->functions[gvars[0]];
    ICHECK(new_func->IsInstance<tir::PrimFuncNode>());
    tir::PrimFunc new_pf = Downcast<tir::PrimFunc>(new_func);
    // Inherit the attr from the original primfunc
    return tir::PrimFunc(new_pf->params, new_pf->body, new_pf->ret_type, new_pf->buffer_map,
                         new_pf->preflattened_buffer_map, f->attrs, new_pf->span);
  }

 private:
  Target target_;
  Array<ObjectRef> config_;
  String work_dir_;
  const runtime::PackedFunc* candgen_func_;
  const tvm::meta_schedule::Database& database_;
};

class MetaScheduleAHB {
 public:
  explicit MetaScheduleAHB(const tvm::meta_schedule::Database& db, Target target)
      : db_(db), target_(target) {}
  IRModule Apply(IRModule mod) {
    IRModule ret_mod_ = IRModule();
    tvm::meta_schedule::ApplyHistoryBest ahb(db_, nullptr, nullptr);
    for (auto& p : mod->functions) {
      GlobalVar gv = p.first;
      BaseFunc func = p.second;
      BaseFunc newfunc = func;
      if (func->IsInstance<tir::PrimFuncNode>()) {
        IRModule tir_mod(Map<GlobalVar, BaseFunc>({{gv, func}}));
        ObjectRef res =
            ahb->Query(gv->name_hint, mod, target_, Array<IRModule>{tir_mod}, nullptr, nullptr);
        // replace the tir func only when the schedule is found in tuning database.
        if (res.defined()) {
          IRModule newmod = Downcast<IRModule>(res);
          ICHECK_EQ(newmod->functions.size(), 1);
          newfunc = (*newmod->functions.begin()).second;
        }
      }

      ret_mod_->Add(gv, newfunc);
    }
    return ret_mod_;
  }

 private:
  const tvm::meta_schedule::Database& db_;
  Target target_;
};

namespace transform {

Pass MetaScheduleTuneIRMod(Target target, Array<ObjectRef> config, String work_dir,
                           const tvm::meta_schedule::Database& database) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext ctx) {
    return MetaScheduleTuner(target, config, work_dir, database).TuneIRMod(m, ctx);
  };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "MetaScheduleTuneIRModule",
                          /*required*/ {},
                          /*traceable*/ true);
}

Pass MetaScheduleTuneTIR(Target target, Array<ObjectRef> config, String work_dir,
                         const tvm::meta_schedule::Database& database) {
  runtime::TypedPackedFunc<tir::PrimFunc(tir::PrimFunc, IRModule, PassContext)> pass_func =
      [=](tir::PrimFunc f, IRModule mod, PassContext ctx) {
        return MetaScheduleTuner(target, config, work_dir, database).TuneTIR(f, ctx);
      };
  return tir::transform::CreatePrimFuncPass(/*pass function*/ pass_func, /*opt level*/ 0,
                                            /*pass name*/ "MetaScheduleTuneTIR",
                                            /*required*/ {},
                                            /*traceable*/ true);
}

Pass MetaScheduleApplyHistoryBest(Target target, Optional<String> work_dir,
                                  Optional<tvm::meta_schedule::Database> database) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext ctx) {
    if (!database.defined()) {
      ICHECK(work_dir.defined());
      String path_workload = work_dir.value() + "/database_workload.json";
      String path_tuning_record = work_dir.value() + "/database_tuning_record.json";
      LOG(INFO) << "Creating JSONDatabase. Workload at: " << path_workload
                << ", Tuning records at: " << path_tuning_record;
      return MetaScheduleAHB(
                 meta_schedule::Database::JSONDatabase(path_workload, path_tuning_record, true),
                 target)
          .Apply(m);
    } else {
      return MetaScheduleAHB(database.value(), target).Apply(m);
    }
  };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "MetaScheduleApplyHistoryBest",
                          /*required*/ {});
}

TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleTuneIRMod").set_body_typed(MetaScheduleTuneIRMod);
TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleTuneTIR").set_body_typed(MetaScheduleTuneTIR);
TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleApplyHistoryBest")
    .set_body_typed(MetaScheduleApplyHistoryBest);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
