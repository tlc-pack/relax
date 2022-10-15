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
#include <tvm/meta_schedule/database.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/tuning_api.h>
#include <tvm/tir/transform.h>

#include "../../printer/text_printer.h"

namespace tvm {
namespace relax {
namespace transform {

Pass MetaScheduleApplyDatabase() {
  using tvm::meta_schedule::Database;
  Target target = Target::Current(false);
  Database database = Database::Current().value();
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext ctx) {
    Map<GlobalVar, BaseFunc> result;
    for (const auto& iter : mod->functions) {
      GlobalVar gv = iter.first;
      BaseFunc base_func = iter.second;
      if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
        if (Optional<tir::Schedule> sch = database->QuerySchedule(
                IRModule({{gv, GetRef<tir::PrimFunc>(prim_func)}}), target, gv->name_hint)) {
          IRModule new_mod = sch.value()->mod();
          ICHECK_EQ(new_mod->functions.size(), 1);
          BaseFunc new_base_func = (*new_mod->functions.begin()).second;
          result.Set(gv, new_base_func);
          continue;
        }
      }
      result.Set(gv, base_func);
    }
    return IRModule(result);
  };
  return CreateModulePass(pass_func, 0, "MetaScheduleApplyDatabase", {});
}

TVM_REGISTER_GLOBAL("relax.transform.MetaScheduleApplyDatabase")
    .set_body_typed(MetaScheduleApplyDatabase);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
