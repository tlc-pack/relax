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
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

bool HasInt64Shape(PrimFunc func) {
  for (const auto& kv : func->buffer_map) {
    for (const PrimExpr& dim_length : kv.second->shape) {
      if (dim_length->dtype.bits() == 64) {
        return true;
      }
    }
  }
  return false;
}

class DataTypePromoter : public StmtExprMutator {
 public:
  static PrimFunc Promote(PrimFunc func) {
    DataTypePromoter promoter;

    // Step 1. Update the buffer map.
    Map<Var, Buffer> buffer_map;
    for (const auto& kv : func->buffer_map) {
      buffer_map.Set(kv.first, promoter.PromoteBuffer(kv.second));
    }

    // Step 2. Mutate the function body.
    Stmt body = promoter(func->body);

    // Step 3. Update the parameters in case some parameters are used in the shapes and strides of
    // the buffers in the buffer map.
    Array<Var> params;
    params.reserve(func->params.size());
    for (const Var& param : func->params) {
      auto it = promoter.var_map_.find(param);
      params.push_back(it == promoter.var_map_.end() ? param : (*it).second);
    }

    PrimFuncNode* p_new_func = func.CopyOnWrite();
    p_new_func->buffer_map = std::move(buffer_map);
    p_new_func->params = std::move(params);
    p_new_func->body = std::move(body);
    return GetRef<PrimFunc>(p_new_func);
  }

 private:
  Stmt VisitStmt_(const ForNode* loop) final {
    Var loop_var = PromoteVar(loop->loop_var);
    PrimExpr min = PromoteValue(loop->min);
    PrimExpr extent = PromoteValue(loop->extent);
    Stmt body = VisitStmt(loop->body);
    Optional<IterVar> thread_binding = NullOpt;
    if (loop->thread_binding.defined()) {
      thread_binding = PromoteIterVar(loop->thread_binding.value());
    }

    if (loop_var.same_as(loop->loop_var) && min.same_as(loop->min) &&
        extent.same_as(loop->extent) && body.same_as(loop->body) &&
        thread_binding.same_as(loop->thread_binding)) {
      return GetRef<Stmt>(loop);
    } else {
      auto n = CopyOnWrite(loop);
      n->loop_var = std::move(loop_var);
      n->min = std::move(min);
      n->extent = std::move(extent);
      n->body = std::move(body);
      n->thread_binding = std::move(thread_binding);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    // Step 1. Promote IterVars
    Array<IterVar> iter_vars;
    iter_vars.reserve(block->iter_vars.size());
    bool iter_var_unchanged = true;
    for (const IterVar& iter_var : block->iter_vars) {
      IterVar new_iter = PromoteIterVar(iter_var);
      iter_vars.push_back(new_iter);
      iter_var_unchanged = iter_var_unchanged && new_iter.same_as(iter_var);
    }

    // Step 2. Promote allocated buffers
    Array<Buffer> alloc_buffers;
    alloc_buffers.reserve(block->alloc_buffers.size());
    bool alloc_buffer_unchanged = true;
    for (const Buffer& buffer : block->alloc_buffers) {
      Buffer new_buf = PromoteBuffer(buffer);
      alloc_buffers.push_back(new_buf);
      alloc_buffer_unchanged = alloc_buffer_unchanged && new_buf.same_as(buffer);
    }

    // Step 3. Promote buffer subregion matches
    Array<MatchBufferRegion> match_buffers;
    match_buffers.reserve(block->match_buffers.size());
    bool match_buffer_unchanged = true;
    for (const MatchBufferRegion& match_buf : block->match_buffers) {
      Buffer buf = PromoteBuffer(match_buf->buffer);
      BufferRegion source = PromoteBufferRegions({match_buf->source})[0];
      if (buf.same_as(match_buf->buffer) && source.same_as(match_buf->source)) {
        match_buffers.push_back(match_buf);
      } else {
        match_buffers.push_back(MatchBufferRegion(buf, source));
        match_buffer_unchanged = false;
      }
    }

    // Step 4. Promote read/write regions
    Array<BufferRegion> reads = PromoteBufferRegions(block->reads);
    Array<BufferRegion> writes = PromoteBufferRegions(block->writes);

    // Step 5. Mutate the block body and block init
    Optional<Stmt> init = NullOpt;
    if (block->init.defined()) {
      init = VisitStmt(block->init.value());
    }
    Stmt body = VisitStmt(block->body);

    if (iter_var_unchanged && reads.same_as(block->reads) && writes.same_as(block->writes) &&
        alloc_buffer_unchanged && match_buffer_unchanged && body.same_as(block->body) &&
        init.same_as(block->init)) {
      return GetRef<Block>(block);
    } else {
      auto n = CopyOnWrite(block);
      n->iter_vars = std::move(iter_vars);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->alloc_buffers = std::move(alloc_buffers);
      n->match_buffers = std::move(match_buffers);
      n->body = std::move(body);
      n->init = std::move(init);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* buf_store) final {
    Buffer buffer = PromoteBuffer(buf_store->buffer);
    PrimExpr value = VisitExpr(buf_store->value);
    Array<PrimExpr> indices = PromoteIndices(buf_store->indices);
    if (buffer.same_as(buf_store->buffer) && value.same_as(buf_store->value) &&
        indices.same_as(buf_store->indices)) {
      return GetRef<BufferStore>(buf_store);
    } else {
      auto n = CopyOnWrite(buf_store);
      n->buffer = std::move(buffer);
      n->value = std::move(value);
      n->indices = std::move(indices);
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* buf_load) final {
    Buffer buffer = PromoteBuffer(buf_load->buffer);
    Array<PrimExpr> indices = PromoteIndices(buf_load->indices);
    if (buffer.same_as(buf_load->buffer) && indices.same_as(buf_load->indices)) {
      return GetRef<BufferLoad>(buf_load);
    } else {
      ObjectPtr<BufferLoadNode> p_buf_load = make_object<BufferLoadNode>(*buf_load);
      p_buf_load->buffer = std::move(buffer);
      p_buf_load->indices = std::move(indices);
      return PrimExpr(p_buf_load);
    }
  }

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC) \
  PrimExpr VisitExpr_(const OP* op) {                     \
    PrimExpr a = this->VisitExpr(op->a);                  \
    PrimExpr b = this->VisitExpr(op->b);                  \
    if (a.same_as(op->a) && b.same_as(op->b)) {           \
      return GetRef<PrimExpr>(op);                        \
    } else {                                              \
      return FUNC(a, b);                                  \
    }                                                     \
  }

  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
  DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = var_map_.find(GetRef<Var>(var));
    return it == var_map_.end() ? GetRef<Var>(var) : (*it).second;
  }

  IterVar PromoteIterVar(IterVar iter) {
    Var var = PromoteVar(iter->var);
    Range dom = PromoteRanges({iter->dom})[0];
    if (var.same_as(iter->var) && dom.same_as(iter->dom)) {
      return iter;
    } else {
      IterVarNode* p_iter = iter.CopyOnWrite();
      p_iter->var = std::move(var);
      p_iter->dom = std::move(dom);
      return GetRef<IterVar>(p_iter);
    }
  }

  Array<BufferRegion> PromoteBufferRegions(Array<BufferRegion> buffer_regions) {
    Array<BufferRegion> new_regions;
    new_regions.reserve(buffer_regions.size());
    bool region_unchanged = true;
    for (const BufferRegion& buf_region : buffer_regions) {
      Buffer buf = PromoteBuffer(buf_region->buffer);
      Array<Range> region = PromoteRanges(buf_region->region);
      if (buf.same_as(buf_region->buffer) && region.same_as(buf_region->region)) {
        new_regions.push_back(buf_region);
      } else {
        new_regions.push_back(BufferRegion(buf, region));
        region_unchanged = false;
      }
    }
    return region_unchanged ? std::move(buffer_regions) : std::move(new_regions);
  }

  Array<Range> PromoteRanges(Array<Range> ranges) {
    Array<Range> new_ranges;
    new_ranges.reserve(ranges.size());
    bool range_unchanged = true;
    for (const Range& range : ranges) {
      PrimExpr min = PromoteValue(range->min);
      PrimExpr extent = PromoteValue(range->extent);
      if (min.same_as(range->min) && extent.same_as(range->extent)) {
        new_ranges.push_back(range);
      } else {
        new_ranges.push_back(Range::FromMinExtent(min, extent));
        range_unchanged = false;
      }
    }

    return range_unchanged ? std::move(ranges) : std::move(new_ranges);
  }

  Array<PrimExpr> PromoteIndices(Array<PrimExpr> indices) {
    Array<PrimExpr> new_indices;
    new_indices.reserve(indices.size());
    bool index_unchanged = true;
    for (const PrimExpr& index : indices) {
      ICHECK(index->dtype.is_int());
      PrimExpr new_index = PromoteValue(index);
      new_indices.push_back(new_index);
      index_unchanged = index_unchanged && new_index.same_as(index);
    }
    return index_unchanged ? indices : new_indices;
  }

  Buffer PromoteBuffer(Buffer buf) {
    auto it = buf_map_.find(buf);
    if (it != buf_map_.end()) {
      return (*it).second;
    }

    auto f_mutate = [this](Array<PrimExpr> expr_array) {
      Array<PrimExpr> new_array;
      new_array.reserve(expr_array.size());
      bool unchanged = true;
      for (const PrimExpr& e : expr_array) {
        PrimExpr new_expr = PromoteValue(e);
        new_array.push_back(new_expr);
        unchanged = unchanged && new_expr.same_as(e);
      }
      return unchanged ? std::move(expr_array) : std::move(new_array);
    };

    Array<PrimExpr> new_shape = f_mutate(buf->shape);
    Array<PrimExpr> new_strides = f_mutate(buf->strides);

    if (new_shape.same_as(buf->shape) && new_strides.same_as(buf->strides)) {
      return buf;
    } else {
      ObjectPtr<BufferNode> p_new_buf = make_object<BufferNode>(*buf.get());
      p_new_buf->shape = std::move(new_shape);
      p_new_buf->strides = std::move(new_strides);

      Buffer new_buf(p_new_buf);
      buf_map_.Set(buf, new_buf);
      return new_buf;
    }
  }

  PrimExpr PromoteValue(PrimExpr value) {
    value = VisitExpr(value);
    if (value->dtype == DataType::Int(64)) {
      return value;
    }

    Optional<PrimExpr> int_value = PromoteInteger(value);
    if (int_value.defined()) {
      return int_value.value();
    }

    const auto* var_value = value.as<VarNode>();
    CHECK(var_value != nullptr) << "ValueError: The input value should be either a constant "
                                   "integer or a variable of integer type";
    auto it_var = this->var_map_.find(GetRef<Var>(var_value));
    return it_var != this->var_map_.end() ? (*it_var).second : PromoteVar(GetRef<Var>(var_value));
  }

  Optional<PrimExpr> PromoteInteger(const PrimExpr& value) {
    const int64_t* int_value = as_const_int(value);
    if (int_value == nullptr) {
      return NullOpt;
    }
    return value->dtype == DataType::Int(64) ? value : make_const(DataType::Int(64), *int_value);
  }

  Var PromoteVar(Var var) {
    if (var->dtype == DataType::Int(64)) {
      return var;
    }

    ObjectPtr<VarNode> p_new_var = make_object<VarNode>(*var.get());
    p_new_var->dtype = DataType::Int(64);
    Var new_var(p_new_var);
    var_map_.Set(var, new_var);
    return new_var;
  }

  Map<Var, Var> var_map_;
  Map<Buffer, Buffer> buf_map_;
};

PrimFunc PromoteDataType(PrimFunc func) {
  // Only apply this pass to TIR that is not from TE schedules
  String name = func->GetAttr<String>("global_symbol").value();
  if (!HasInt64Shape(func)) {
    return func;
  }
  return DataTypePromoter::Promote(std::move(func));
}

namespace transform {

Pass PromoteDataType() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return PromoteDataType(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.PromoteDataType", {});
}

TVM_REGISTER_GLOBAL("tir.transform.PromoteDataType").set_body_typed(PromoteDataType);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
