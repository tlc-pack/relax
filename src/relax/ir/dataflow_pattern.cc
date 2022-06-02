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
 * \file src/tvm/relax/ir/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relax (inherited from Relay).
 */

#include <tvm/relax/dataflow_pattern.h>

namespace tvm {
namespace relax {

ExternFuncPattern::ExternFuncPattern(String global_symbol) {
  ObjectPtr<ExternFuncPatternNode> n = make_object<ExternFuncPatternNode>();
  n->global_symbol_ = std::move(global_symbol);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExternFuncPatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.ExternFuncPattern")
    .set_body_typed([](String global_symbol) { return ExternFuncPattern(global_symbol); });

VarPattern::VarPattern(String name_hint) {
  ObjectPtr<VarPatternNode> n = make_object<VarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(VarPatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.VarPattern").set_body_typed([](String name_hint) {
  return VarPattern(name_hint);
});

TVM_REGISTER_NODE_TYPE(DataflowVarPatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.DataflowVarPattern")
    .set_body_typed([](String name_hint) { return DataflowVarPattern(name_hint); });

DynTensorTypePattern::DynTensorTypePattern(DFPattern pattern, DynTensorType type) {
  ObjectPtr<DynTensorTypePatternNode> n = make_object<DynTensorTypePatternNode>();
  n->pattern = std::move(pattern);
  n->type = std::move(type);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DynTensorTypePatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.DynTensorTypePattern")
    .set_body_typed([](DFPattern pattern, DynTensorType type) {
      return DynTensorTypePattern(pattern, type);
    });

TVM_REGISTER_NODE_TYPE(RuntimeDepShapePatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.RuntimeDepShapePattern").set_body_typed([] {
  return RuntimeDepShapePattern(make_object<RuntimeDepShapePatternNode>());
});

}  // namespace relax
}  // namespace tvm
