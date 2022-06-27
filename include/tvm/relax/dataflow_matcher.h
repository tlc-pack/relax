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
 * \file tvm/relax/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAX_DATAFLOW_MATCHER_H_
#define TVM_RELAX_DATAFLOW_MATCHER_H_

#include <tvm/relax/dataflow_pattern.h>
#include <tvm/runtime/container/optional.h>

namespace tvm {
namespace relax {

/**
 * \brief Determine if a pattern matches an expression
 *
 * \param pattern The pattern to match
 * \param expr The expression to match
 * \param var2val The mapping from relax.Var to relax.Expr
 * \return true if matched
 * \return false if unmatched
 */
bool MatchExprPattern(DFPattern pattern, Expr expr,
                      Optional<runtime::Map<Var, Expr>> var2val = NullOpt);

/**
 * \brief Graph-wise pattern matcher to return node maps (pattern -> expr). This algorithm returns
 * the first matched sub-graph. Use `start_hint` to specify the starting point of the matching so
 * that we can distinguish multiple matches.
 *
 * \param gpatterns The graph-wise patterns.
 * \param fn The function to match.
 * \param start_hint The starting point expression to match to distinguish multiple matches.
 * \return tvm::runtime::Map<DFPattern, VarBinding>
 */
tvm::runtime::Map<DFPattern, VarBinding> MatchGraphPattern(std::shared_ptr<GraphPattern> gpatterns,
                                                           const Function& fn,
                                                           Optional<Expr> start_hint = NullOpt);

/** \brief Call MatchGraphPattern through pattern's graph constraints */
inline tvm::runtime::Map<DFPattern, VarBinding> MatchGraphPattern(
    DFPattern pattern, const Function& fn, Optional<Expr> start_hint = NullOpt) {
  ICHECK(nullptr != pattern->graph_constraint)
      << "Graph constraints are required to match graph patterns";
  return MatchGraphPattern(pattern->graph_constraint, fn, start_hint);
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DATAFLOW_MATCHER_H_
