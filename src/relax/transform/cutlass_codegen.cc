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
#include <tvm/ir/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

const static constexpr char* kCutlassKernel = "cutlass_kernel";
const static constexpr char* kCutlassCodegen = "cutlass_codegen";
const static constexpr char* kCSource = "c_source";
const static constexpr char* kCSourceFmt = "c_source_fmt";
const static constexpr char* kCSourceFmtCuda = "cu";

void StringReplace(std::string* subject, const std::string& search, const std::string& replace) {
  for (size_t pos = 0; (pos = subject->find(search, pos)) != std::string::npos;
       pos += replace.length()) {
    subject->replace(pos, search.length(), replace);
  }
}

ExternFunc CodegenWithCutlass(const tir::PrimFuncNode* pf, String global_symbol) {
  using namespace tvm::tir;
  Optional<Array<runtime::String>> cutlass_op =
      pf->attrs.GetAttr<Array<runtime::String>>(kCutlassKernel);
  ICHECK(cutlass_op.defined()) << "No cutlass kernel is specified";
  auto f = tvm::runtime::Registry::Get("tvm.relax.cutlass.get_graph_pattern_code");
  ICHECK(f != nullptr) << "Cannot find cutlass codegen function";
  std::string source = (*f)(cutlass_op.value());
  StringReplace(&source, "{global_symbol}", global_symbol);
  ExternFunc ret(global_symbol);
  ret = WithAttrs(std::move(ret), Map<String, ObjectRef>{
                                      {String(kCSource), String(source)},
                                      {String(kCSourceFmt), String(kCSourceFmtCuda)},
                                  });
  return ret;
}

namespace transform {

Pass CutlassCodegen() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) -> IRModule {
    IRModuleNode* mod = m.CopyOnWrite();
    Map<GlobalVar, BaseFunc> functions;
    for (const auto& kv : mod->functions) {
      GlobalVar gv = kv.first;
      BaseFunc base_func = kv.second;
      if (const tir::PrimFuncNode* pf = base_func.as<tir::PrimFuncNode>()) {
        if (Optional<Integer> _ = pf->attrs.GetAttr<Integer>(kCutlassCodegen)) {
          functions.Set(gv, CodegenWithCutlass(pf, gv->name_hint));
          continue;
        }
      }
      functions.Set(gv, kv.second);
    }
    mod->functions = std::move(functions);
    return GetRef<IRModule>(mod);
  };
  return CreateModulePass(pass_func, 0, "CutlassCodegen", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CutlassCodegen").set_body_typed(CutlassCodegen);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
