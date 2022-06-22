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
#ifndef TVM_RELAX_DATAFLOW_GRAPH_CONSTRAINT_H_
#define TVM_RELAX_DATAFLOW_GRAPH_CONSTRAINT_H_

#include <tvm/relax/dataflow_pattern.h>

#include <initializer_list>
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
  inline void add_constraint(const DFPatternNode* src, const DFPatternNode* dst, Cons... cons) {
    static_assert(sizeof...(cons) > 0, "Constraints should not be empty!");
    auto& src_cons = constraints[src];
    src_cons.reserve(src_cons.size() + sizeof...(cons));
    // same as `(src_cons.push_back(cons), ...)` in C++17.
    (void)std::initializer_list<int>{(src_cons.emplace_back(dst, cons), 0)...};
  }
  // special constraints.
  enum ExternUse { kMay, kMust, kMustNot } allow_extern_use = kMay;
  // src node -> <dst node, constraint type> constraints.
  std::map<const DFPatternNode*, std::vector<std::pair<const DFPatternNode*, PairCons>>>
      constraints;
};
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DATAFLOW_GRAPH_CONSTRAINT_H_
