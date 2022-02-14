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
 * \file tvm/relax/transform/meta_schedule_ahb.cc
 * \brief Pass for applying the best schedule from tuning database.
 */

#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class MetaScheduleAHB {
 public:
  explicit MetaScheduleAHB(IRModule mod, const tvm::meta_schedule::Database& db, Target target)
      : mod_(mod), db_(db), target_(target) {}
  IRModule Apply() {
    ret_mod_ = IRModule();
    tvm::meta_schedule::ApplyHistoryBest ahb(db_);
    for (auto& p : mod_->functions) {
      GlobalVar gv = p.first;
      BaseFunc func = p.second;
      BaseFunc newfunc = func;
      if (func->IsInstance<tir::PrimFuncNode>()) {
        IRModule tir_mod(Map<GlobalVar, BaseFunc>({{gv, func}}));
        ObjectRef res = ahb->Query(gv->name_hint, mod_, target_, Array<IRModule>{tir_mod});
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
  IRModule mod_;
  const tvm::meta_schedule::Database& db_;
  Target target_;
  IRModule ret_mod_;
};

namespace transform {

Pass MetaScheduleApplyHistoryBest(const tvm::meta_schedule::Database& database, Target target) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return MetaScheduleAHB(m, database, target).Apply(); };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "MetaScheduleApplyHistoryBest",
                          /*required*/ {});
}

TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleApplyHistoryBest")
    .set_body_typed(MetaScheduleApplyHistoryBest);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
