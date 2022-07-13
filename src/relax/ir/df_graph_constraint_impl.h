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
 * \file src/relax/ir/dataflow_graph_constraint.h
 * \brief Describing and matching graph-wise patterns.
 */
#ifndef TVM_RELAX_IR_DF_GRAPH_CONSTRAINT_IMPL_H_
#define TVM_RELAX_IR_DF_GRAPH_CONSTRAINT_IMPL_H_

#include <tvm/relax/dataflow_pattern.h>

#include <initializer_list>
#include <map>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

struct PairCons {
  enum Type {
    kUsedBy,
    kOnlyUsedBy,
  } type = kUsedBy;
  int index = -1; /* means whatever */
  inline explicit PairCons(Type t, int index = -1) : type(t), index(index) {}
};

struct GraphPattern final {
  template <typename... Cons>
  inline void add_constraint(const DFPatternNode* def, const DFPatternNode* use, PairCons cons) {
    constraints[def].emplace(use, cons);
  }
  // special constraints.
  enum ExternUse { kMay, kMust, kMustNot } allow_extern_use = kMay;
  // src node -> <dst node, constraint type> constraints.
  std::map<const DFPatternNode*, std::map<const DFPatternNode*, PairCons>> constraints;
};
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DF_GRAPH_CONSTRAINT_IMPL_H_