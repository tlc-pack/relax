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

#include <tvm/relax/analysis.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

using namespace tir;

class PatternKindAnalyzer : public StmtExprVisitor {
 public:
  explicit PatternKindAnalyzer(const PrimFunc& func) {
    for (const Var& param : func->params) {
      param_buffers_.insert(func->buffer_map.Get(param).value());
    }
  }

 private:
  bool IsOutputBlock(const BlockNode* block) {
    for (const BufferRegion& write_region : block->writes) {
      if (param_buffers_.count(write_region->buffer)) {
        return true;
      }
    }
    return false;
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    ICHECK(!store_.defined());
    store_ = GetRef<BufferStore>(op);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->indices.size() > 0) {
      // zero-rank buffer loads are regarded as constant, skip it.
      // E.g. A[i] = B[i] + C[()] will be element-wise rather than broadcast or injective.
      loads_.push_back(GetRef<BufferLoad>(op));
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      // Skip the root block
      StmtVisitor::VisitStmt(op->body);
      return;
    }

    // Step 1. Clear loads and store
    loads_.clear();
    store_ = NullOpt;
    // Step 2. Visit block body.
    StmtVisitor::VisitStmt(op->body);
    BufferStore store = store_.value();

    // Step 3. Checking load store indices pattern
    relay::OpPatternKind index_pair_pattern = relay::kElemWise;
    if (loads_.empty()) {
      // E.g. A[i] = B[()] or A[i] = 1
      index_pair_pattern = relay::kBroadcast;
    } else {
      for (const BufferLoad& load : loads_) {
        // Since elemwise is stricter than broadcast and broadcast is stricter than injective,
        // while the order amount enums: kElemWise < kBroadcast < kInjective.
        // We can simpily use `std::max` to detect these three patterns.
        // E.g Here is only one store node but two load nodes, like C[i, j] = A[i, j] + B[i]
        // Buffer C and A are elemwise but C and B are broadcast. So the whole block follows
        // broadcast pattern.
        if (IsElemwisePattern(store, load)) {
          index_pair_pattern = std::max(index_pair_pattern, relay::kElemWise);
        } else if (IsBroadcastPattern(store, load)) {
          index_pair_pattern = std::max(index_pair_pattern, relay::kBroadcast);
        } else if (IsInjectivePattern(store, load)) {
          index_pair_pattern = std::max(index_pair_pattern, relay::kInjective);
        } else {
          index_pair_pattern = relay::kOpaque;
          break;
        }
      }
    }
    // If the block index pattern is not opaque, update kind.
    if (index_pair_pattern != relay::kOpaque) {
      // This rule for softmax: reduce + injective.
      if (IsOutputBlock(op) && kind_ == relay::kCommReduce) {
        kind_ = relay::kOutEWiseFusable;
      } else {
        kind_ = std::max(kind_, index_pair_pattern);
      }
      return;
    }

    // Step 4. Checking if the block contains reduce axis by looking into block iterators.
    bool has_reduction = false;
    Array<Var> reduce_vars;
    for (const IterVar& it : op->iter_vars) {
      if (it->iter_type == kCommReduce) {
        has_reduction = true;
        reduce_vars.push_back(it->var);
      }
    }

    if (has_reduction) {
      if (IsFMA(op->body)) {
        // FMA is regards as kOutEWiseFusable, e.g. Matmul or Conv.
        kind_ = std::max(kind_, relay::kOutEWiseFusable);
        return;
      } else {
        for (size_t i = 0; i < loads_.size(); ++i) {
          // If it's not a pure reduce, regards as kOutEWiseFusable.
          // This rule works for pooling for now.
          if (!IsPureReducePattern(reduce_vars, loads_[i]->indices)) {
            kind_ = std::max(kind_, relay::kOutEWiseFusable);
            return;
          }
        }
      }
      kind_ = std::max(kind_, relay::kCommReduce);
    } else {
      kind_ = relay::kOpaque;
    }
  }

  /********** Helper Functions **********/

  /*! \brief Checking if two arrays contains same elements. */
  static bool IsSameArray(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!lhs[i].same_as(rhs[i])) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Checking the load indices and store indices follows elemwise pattern.
   * It's elemwise pattern iff load indices and store indices are the same.
   * E.g A[i, j] = B[i, j]
   */
  static bool IsElemwisePattern(const BufferStore& store, const BufferLoad& load) {
    return IsSameArray(store->indices, load->indices);
  }

  /*!
   * \brief Checking the load indices and store indices follows broadcast pattern.
   * It's broadcast pattern iff all load indices are in the store indices in order
   * E.g. A[i, j] = B[i] is broadcast since all load indices(`i`) are in the store indices
   *      A[i, j] = B[i, k] is not broadcast since `k` are not in the store indices.
   *      A[i, j] = B[j, i] is not broadcast the load indices are not in the same order as store's
   */
  static bool IsBroadcastPattern(const BufferStore& store, const BufferLoad& load) {
    size_t ndim_load_buf = load->buffer->shape.size();
    size_t ndim_store_buf = store->buffer->shape.size();

    for (size_t i = 0, j = 0; i < ndim_load_buf; ++i) {
      if (is_const_int(load->buffer->shape[i], 1) && is_const_int(load->indices[i], 0)) {
        // Skip unit load dimensions
        // E.g. A[i, j] = B[1, j] is still broadcast
        continue;
      }

      // Try to find the i-th load indice in the store indices.
      while (j < ndim_store_buf && !store->indices[j].same_as(load->indices[i])) {
        ++j;
      }

      // It's not broadcast if we cannot find load indices in the store indices in order.
      if (j == ndim_store_buf) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Checking the load indices and store indices follows injective pattern.
   * It's injective pattern iff all load indice vars are in the store indices, no matter orders.
   * Note that we only support store indices are direct vars so far, which can be enhance later.
   * E.g. A[i, j] = B[j, i] is injective.
   *      A[i, j] = B[i - j] is injective since the load indice vars are only i, j
   */
  static bool IsInjectivePattern(const BufferStore& store, const BufferLoad& load) {
    std::unordered_set<const VarNode*> vars;
    for (const PrimExpr& store_index : store->indices) {
      if (const auto* v = store_index.as<VarNode>()) {
        vars.insert(v);
      } else {
        return false;
      }
    }
    for (const PrimExpr& load_index : load->indices) {
      // return false if there are vars used in load indices but not in store indices.
      if (tir::UsesVar(load_index, [&vars](const VarNode* var) { return !vars.count(var); })) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Checking the load indices and store indices allow data reuse.
   * It allow data reuse iff there is any vars in load indices but they are not in store indices
   * E.g. Store = A[i, j] and Load = B[i, j, k] allow data reuse.
   *      Store = A[i, j] and Load = B[i, j + k] allow data reuse.
   */
  static bool IsAllowReusePattern(const BufferStore& store, const BufferLoad& load) {
    std::unordered_set<const VarNode*> vars;
    for (const PrimExpr& index : store->indices) {
      if (const auto* v = index.as<VarNode>()) {
        vars.insert(v);
      } else {
        return false;
      }
    }
    for (const PrimExpr& index : load->indices) {
      PreOrderVisit(index, [&](const ObjectRef& node) {
        if (const auto* v = node.as<VarNode>()) {
          if (vars.count(v)) {
            vars.erase(v);
          }
        }
        return true;
      });
    }
    return !vars.empty();
  }

  /*! \brief Checking if the stmt is multiply add. E.g. C[i, j] += A[i, k] * B[j, k] */
  static bool IsFMA(const Stmt& body) {
    if (const auto* store = body.as<BufferStoreNode>()) {
      if (const auto* add = store->value.as<AddNode>()) {
        if (const auto* l = add->a.as<BufferLoadNode>()) {
          if (const auto* r = add->b.as<MulNode>()) {
            bool incremental =
                store->buffer.same_as(l->buffer) && IsSameArray(store->indices, l->indices);
            const auto* l_load = r->a.as<BufferLoadNode>();
            const auto* r_load = r->b.as<BufferLoadNode>();
            if (incremental && l_load && r_load) {
              return IsAllowReusePattern(GetRef<BufferStore>(store), GetRef<BufferLoad>(l_load)) &&
                     IsAllowReusePattern(GetRef<BufferStore>(store), GetRef<BufferLoad>(r_load));
            }
          }
        }
      }
    }
    return false;
  }

  /*!
   * \brief Checking if it is pure reduce pattern.
   * It's pure reduce pattern iff all reduces axis are directly reduce var
   * E.g. A[i] = sum(B[i, j]) is pure reduce
   *      A[i] = sum(B[i, j + k]) is not pure reduce
   *      pooling is not pure reduce
   */
  static bool IsPureReducePattern(Array<Var> reduce_loops, Array<PrimExpr> indices) {
    for (const PrimExpr& e : indices) {
      int id = -1;
      if (UsesVar(e, [&](const VarNode* var) {
            for (size_t i = 0; i < reduce_loops.size(); ++i) {
              if (reduce_loops[i].get() == var) {
                id = i;
                return true;
              }
            }
            return false;
          })) {
        if (!reduce_loops[id].same_as(e)) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  /*!
   * \brief The BufferStore node in the current block.
   * \note We only support one BufferStore node in a block (ususally generated by TE compute)
   */
  Optional<BufferStore> store_;
  /*! \brief The BufferLoad nodes in the current block. */
  Array<BufferLoad> loads_;
  /*! \brief The result of op pattern. */
  relay::OpPatternKind kind_ = relay::kElemWise;
  /*! \brief The buffers from function params. I.e. the input and output buffers. */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> param_buffers_;

 public:
  relay::OpPatternKind GetResult() { return kind_; }
};

relay::OpPatternKind AnalyzeOpPatternKind(const PrimFunc& func) {
  PatternKindAnalyzer analyzer(func);
  analyzer(func->body);
  return analyzer.GetResult();
}

}  // namespace relax
}  // namespace tvm
