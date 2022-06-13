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
 * \file src/tvm/relay/dataflow_matcher_impl.h
 * \brief The auxiliary data structure for dataflow matcher.
 */
#ifndef TVM_RELAX_IR_DATAFLOW_MATCHER_IMPL_H_
#define TVM_RELAX_IR_DATAFLOW_MATCHER_IMPL_H_

#include <tvm/arith/analyzer.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/dataflow_pattern_functor.h>

namespace tvm {
namespace relax {

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  using var2val_t = runtime::Map<Var, Expr>;

  explicit DFPatternMatcher(const Expr& root_expr, var2val_t var2val) : var2val_(var2val) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }
  const var2val_t var2val_;

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const OrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AndPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;

  bool VisitDFPattern_(const RuntimeDepShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataflowVarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const GlobalVarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExternFuncPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const PrimArrPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);

  std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  arith::Analyzer analyzer_;
  bool memoize_ = true;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DATAFLOW_MATCHER_IMPL_H_
