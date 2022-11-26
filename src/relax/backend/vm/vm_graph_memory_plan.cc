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
 * \file src/relax/backend/vm/vm_graph_memory_plan.cc
 * \brief Perform memory planning for memory reuse.
 * \note
 *  - Nested functions are not considered yet.
 *  - Symbolic shape cases (ones with MatchShape) are completely not considered yet.
 *  - RuntimeDepShape is not allowed at this moment.
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "../../../relay/transforms/pattern_utils.h"
#include "../../../support/arena.h"
#include "../../op/make_op.h"

namespace tvm {
namespace relax {

/*!
 * \brief A representation of a block of reusable memory required at runtime.
 * \details Only the tensors whose memory can be "possibly reused" will have their storage token. In
 * other words, we do not have storage token for tensor
 * - that is a function parameter,
 * - that is a function return value,
 * - one of whose use site is a BindingBlock different from its allocation site,
 * - that is used as a condition or branch return of a IfNode
 * - that is used as the body of a SeqExprNode.
 *
 * In practice, we do create a storage token for such tensor at first. But at any time we find a
 * tensor satisfying any of the conditions above, we erase its storage token.
 */
struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief Number of bytes, which is impossible to exceed the range of int. */
  int64_t bytes{-1};
  /*! \brief The shape of the tensor */
  Array<PrimExpr> shape;
  /*! \brief The corresponding tensor dtype. */
  DataType dtype;
  /*! \brief The storage id */
  int storage_id{-1};
  /*! \brief The variable corresponding to the allocated storage */
  Var storage{nullptr};

  std::string ToString() const {
    ICHECK(shape.defined());
    std::ostringstream os;
    os << "{storage_id: " << storage_id << ", bytes: " << bytes << ", shape: " << shape
       << ", dtype: " << dtype << ", ref_counter: " << ref_counter << "}";
    return os.str();
  }
};

/*!
 * \brief A data structure used to represent the storage tokens of Exprs.
 * \details
 * Specially,
 * - the tokens of a Tuple is represented as a TokenContainer with `is_tuple` true, `token` nullptr
 * and `filed_tokens` being the list of the TokenContainers of its fields,
 * - the tokens of a TupleGetItem is the TokenContainer at the indicated position of the
 * TokenContainer of its `tuple`.
 *
 * Since not every Expr has a storage token, consider the following case
 *
 *    @R.function
 *    def func(x: R.Tensor((2, 4), "float32")):
 *      y = R.builtin.alloc_tensor((2, 4), "float32")
 *      exp(x, y)
 *      t = (x, y)
 *      return t[1]
 *
 * Here since `x` is a parameter, we do not create a storage token for it. So if the storage tokens
 * of a Tuple is simply represented as a list of the field storage tokens, then the storage tokens
 * of `t` are represented as a list which only contains the storage token of `y`. And consequently,
 * when getting the storage token of `t[1]`, we use 1 to index the list of single element inside,
 * which leads to error. So this is why we introduce TokenContainer as a data structure for Exprs'
 * storage tokens.
 *
 * \note If an Expr has no corresponding storage token, its TokenContainer has `is_tuple` true,
 * empty `field_containers` and null `token`, in the same form as the TokenContainer of an empty
 * Tuple.
 */
struct TokenContainer {
  TokenContainer() : is_tuple(true), field_containers{}, token(nullptr) {}

  TokenContainer(bool is_tuple, std::vector<TokenContainer> field_containers, StorageToken* token) {
    if (is_tuple) {
      ICHECK(token == nullptr);
    } else {
      ICHECK_NOTNULL(token);
      ICHECK(field_containers.empty());
    }
    this->is_tuple = is_tuple;
    this->field_containers = std::move(field_containers);
    this->token = token;
  }

  bool IsEmptyToken() const {
    return is_tuple == true && field_containers.empty() && token == nullptr;
  }

  /*! \brief Apply the given function to each storage token in this TokenContainer */
  void ApplyToTokens(std::function<void(StorageToken*)> f_apply) const {
    if (is_tuple) {
      for (const TokenContainer& field_container : field_containers) {
        field_container.ApplyToTokens(f_apply);
      }
    } else {
      ICHECK_NOTNULL(token);
      f_apply(token);
    }
  }

  /*! \brief Remove the specified token from this TokenContainer */
  void RemoveToken(const StorageToken* token_to_remove) {
    if (is_tuple) {
      for (TokenContainer& container : field_containers) {
        container.RemoveToken(token_to_remove);
      }
    } else if (token == token_to_remove) {
      // Set to an empty token.
      is_tuple = true;
      ICHECK(field_containers.empty());
      token = nullptr;
    }
  }

  std::string ToString() const {
    std::ostringstream os;
    os << "[";
    ApplyToTokens([&os](StorageToken* token) { os << token->ToString() << ", "; });
    os << "]";
    return os.str();
  }

  bool is_tuple;
  std::vector<TokenContainer> field_containers;
  StorageToken* token;
};

/**
 * \brief Memory manager for flattened 1d memory (buffers)
 */
class TokenAllocator1D {
 public:
  /*!
   * \brief Request a storage token from the available token pool for a given prototype.
   * \param prototype The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype) {
    ICHECK_EQ(prototype->storage_id, -1);
    ICHECK_EQ(prototype->bytes, -1);
    ICHECK_GT(prototype->ref_counter, 0);
    ICHECK(!prototype->storage.defined());

    // Calculate the size in byte.
    int64_t size = GetMemorySize(prototype);
    // Search memory blocks in [size / match_range_, size * match_range_)
    auto begin = available_pool_.lower_bound(size / match_range_);
    auto mid = available_pool_.lower_bound(size);
    auto end = available_pool_.upper_bound(size * match_range_);
    // Search for memory block that equals or is larger than the requested size
    for (auto it = mid; it != end; ++it) {
      StorageToken* available_token = it->second;
      ICHECK_EQ(available_token->ref_counter, 0);
      ICHECK_LE(size, available_token->bytes);
      available_token->ref_counter = prototype->ref_counter;
      // find a exact match, erase from map and return
      available_pool_.erase(it);
      return available_token;
    }
    // Then search for memory block that is smaller than the requested size.
    for (auto it = mid; it != begin;) {
      --it;
      StorageToken* available_token = it->second;
      ICHECK_EQ(available_token->ref_counter, 0);
      ICHECK_GE(size, available_token->bytes);
      available_token->bytes = size;
      available_token->ref_counter = prototype->ref_counter;
      // erase from map and return
      available_pool_.erase(it);
      return available_token;
    }
    // Return `nullptr` indicating that no satisfiable storage token is found in the available pool.
    return nullptr;
  }

  /*!
   * \brief Allocate a storage token for the input prototype token
   * \param prototype The prototype token.
   * \param storage_id The id of this token.
   */
  StorageToken* Alloc(StorageToken* prototype, int storage_id) {
    ICHECK_EQ(prototype->storage_id, -1);
    ICHECK(!prototype->storage.defined());
    int64_t size = GetMemorySize(prototype);
    if (prototype->bytes != -1) {
      ICHECK_EQ(size, prototype->bytes);
    } else {
      prototype->bytes = size;
    }
    prototype->storage_id = storage_id;
    full_pool_.push_back(prototype);
    return prototype;
  }

  /*!
   * \brief Release the input token, putting it into the available pool.
   * \param token The token to be released.
   */
  void Release(StorageToken* token) {
    ICHECK_GE(token->storage_id, 0);
    ICHECK_GE(token->bytes, 0);
    ICHECK_EQ(token->ref_counter, 0);
    ICHECK(!token->storage.defined());
    available_pool_.insert({token->bytes, token});
  }

  std::string DumpMemoryAllocation() const {
    std::ostringstream os;
    double total_gb = 0.0;
    os << "=========================== Dump Memory Allocation ===========================\n";
    for (const StorageToken* token : full_pool_) {
      double size = 1.0 * token->bytes / (1ll << 30);
      total_gb += size;
      os << " - Allocated " << size << " GB\n";
    }
    os << "Total allocated memory that are possibly used: " << total_gb << " GB";
    return os.str();
  }

 private:
  /*!
   * \brief Get the size of the consumed memory of a prototype token.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  int64_t GetMemorySize(StorageToken* prototype) {
    ICHECK_EQ(prototype->storage_id, -1);
    if (prototype->bytes != -1) {
      return prototype->bytes;
    }

    int64_t size = 1;
    for (const PrimExpr& dim_len : prototype->shape) {
      const int64_t* p_dim_len = tir::as_const_int(dim_len);
      ICHECK_NOTNULL(p_dim_len);
      size *= *p_dim_len;
    }
    size *= (prototype->dtype.bits() * prototype->dtype.lanes() + 7) / 8;
    prototype->bytes = size;
    return size;
  }

 private:
  // scale used for rough match
  const int match_range_{16};
  // free list of storage entry
  std::multimap<int64_t, StorageToken*> available_pool_;
  // all the storage resources available
  std::vector<StorageToken*> full_pool_;
  /*! \brief Number of storages */
  int n_storage_;
};

/*! \brief The base class for the storage allocation visitor */
class StorageAllocatorBaseVisitor : public ExprVisitor {
 public:
  explicit StorageAllocatorBaseVisitor(support::Arena* arena) : arena_(arena) {}

 protected:
  void VisitBindingBlock_(const BindingBlockNode* block) override {
    block_stack_.push_back(block);
    ExprVisitor::VisitBindingBlock_(block);
    ICHECK(!block_stack_.empty());
    ICHECK(block_stack_.back() == block);
    block_stack_.pop_back();
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    this->VisitVarDef(binding->var);
    const TokenContainer& tokens = GetTokens(binding->value);
    ExprUsesTokens(binding->var.get(), tokens);
  }

  void VisitExpr_(const TupleNode* tuple) override {
    std::vector<TokenContainer> tokens;
    for (const Expr& field : tuple->fields) {
      const TokenContainer& field_containers = GetTokens(field);
      tokens.push_back(field_containers);
    }
    ExprUsesTokens(tuple, TokenContainer(/*is_tuple=*/true,                       //
                                         /*field_containers=*/std::move(tokens),  //
                                         /*token=*/nullptr));
  }

  void VisitExpr_(const TupleGetItemNode* tuple_item) override {
    const TokenContainer& container = GetTokens(tuple_item->tuple);
    ICHECK(container.is_tuple);
    if (static_cast<int>(container.field_containers.size()) > tuple_item->index) {
      ICHECK_GE(tuple_item->index, 0);
      ExprUsesTokens(tuple_item, container.field_containers[tuple_item->index]);
    } else {
      ICHECK(container.IsEmptyToken());
      token_map_[tuple_item] = no_tokens_;
    }
  }

  // The function is the place where recursive visit happens.
  const TokenContainer& GetTokens(const Expr& expr) {
    this->VisitExpr(expr);
    auto it = token_map_.find(expr.get());
    if (it == token_map_.end()) {
      token_map_[expr.get()] = no_tokens_;
      return no_tokens_;
    }
    return it->second;
  }

  void ExprUsesTokens(const ExprNode* expr, const TokenContainer& container) {
    token_map_[expr] = container;
  }

  /*! \brief The allocator */
  support::Arena* arena_;
  /*! \brief The mapping from each Expr to its corresponding storage tokens */
  std::unordered_map<const ExprNode*, TokenContainer> token_map_;
  /*! \brief The binding block stack */
  std::vector<const BindingBlockNode*> block_stack_;
  /*! \brief An empty token map */
  const TokenContainer no_tokens_;
};

class StorageAllocatorInit : public StorageAllocatorBaseVisitor {
 public:
  explicit StorageAllocatorInit(support::Arena* arena) : StorageAllocatorBaseVisitor(arena) {}

  std::unordered_map<const ExprNode*, TokenContainer> Initialize(const Function& func) {
    for (const Var& param : func->params) {
      token_map_[param.get()] = no_tokens_;
      this->VisitVarDef(param);
    }
    const TokenContainer& body_tokens = GetTokens(func->body);
    // Erase the tokens of the function output.
    body_tokens.ApplyToTokens([this](StorageToken* token) { this->EraseToken(token); });
    return this->token_map_;
  }

 private:
  void VisitExpr_(const VarNode* var) final { ICHECK(token_map_.count(var)); }

  void VisitExpr_(const CallNode* call) final {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op == alloc_tensor_op) {
      this->CreateToken(call);
      return;
    }

    // `block_stack_` here is possibly empty (e.g., the function body is a `call_packed`).
    const BindingBlockNode* cur_block = block_stack_.empty() ? nullptr : block_stack_.back();
    for (const Expr& arg : call->args) {
      const TokenContainer& container = GetTokensWithAllocSiteCheck(arg, cur_block);
      IncreaseRefCounter(container);
    }
  }

  void VisitExpr_(const IfNode* if_node) final {
    const TokenContainer& cond_tokens = GetTokens(if_node->cond);
    const TokenContainer& then_tokens = GetTokens(if_node->true_branch);
    const TokenContainer& else_tokens = GetTokens(if_node->false_branch);
    cond_tokens.ApplyToTokens([this](StorageToken* token) { this->EraseToken(token); });
    then_tokens.ApplyToTokens([this](StorageToken* token) { this->EraseToken(token); });
    else_tokens.ApplyToTokens([this](StorageToken* token) { this->EraseToken(token); });
  }

  void VisitExpr_(const SeqExprNode* seq) final {
    for (const BindingBlock& binding_block : seq->blocks) {
      this->VisitBindingBlock(binding_block);
    }
    const TokenContainer& body_tokens = GetTokens(seq->body);
    body_tokens.ApplyToTokens([this](StorageToken* token) { this->EraseToken(token); });
  }

  TokenContainer CreateToken(const CallNode* call) {
    ICHECK(!token_map_.count(call));

    // The current implementation guarantees that the input can only have DynTensorType.
    const auto* ttype = call->checked_type().as<DynTensorTypeNode>();
    const auto* shape = call->shape().as<ShapeExprNode>();
    const auto* attrs = call->attrs.as<AllocTensorAttrs>();
    ICHECK_NOTNULL(ttype);
    ICHECK_NOTNULL(attrs);
    ICHECK(call->shape().same_as(call->args[0]));
    ICHECK(!ttype->IsUnknownDtype());
    ICHECK(ttype->dtype == attrs->dtype);

    if (ttype->IsUnknownNdim() || shape == nullptr) {
      token_map_[call] = no_tokens_;
      return no_tokens_;
    }
    // Does not support planning for symbolic shape at this moment.
    for (const PrimExpr& dim_len : shape->values) {
      if (!tir::is_const_int(dim_len)) {
        token_map_[call] = no_tokens_;
        return no_tokens_;
      }
    }

    auto* token = arena_->make<StorageToken>();
    token->dtype = ttype->dtype;
    token->shape = shape->values;
    ICHECK(!block_stack_.empty());
    token2block_[token] = block_stack_.back();

    TokenContainer token_container(/*is_tuple=*/false, /*field_containers=*/{}, token);
    ExprUsesTokens(call, token_container);
    return token_container;
  }

  void ExprUsesTokens(const ExprNode* expr, const TokenContainer& container) {
    StorageAllocatorBaseVisitor::ExprUsesTokens(expr, container);
    container.ApplyToTokens(
        [this, expr](StorageToken* token) { this->token2exprs_[token].push_back(expr); });
  }

  const TokenContainer& GetTokensWithAllocSiteCheck(const Expr& expr,
                                                    const BindingBlockNode* cur_block) {
    const TokenContainer& container = GetTokens(expr);
    container.ApplyToTokens([this, cur_block](StorageToken* token) {
      auto it = this->token2block_.find(token);
      ICHECK(it != this->token2block_.end());
      if (it->second != cur_block) {
        EraseToken(token);
      }
    });
    return token_map_[expr.get()];
  }

  void IncreaseRefCounter(const TokenContainer& token_container) {
    token_container.ApplyToTokens([](StorageToken* token) { token->ref_counter += 1; });
  }

  void EraseToken(StorageToken* token) {
    const std::vector<const ExprNode*>& exprs = token2exprs_[token];
    for (const ExprNode* expr : exprs) {
      token_map_[expr].RemoveToken(token);
    }
    token2exprs_.erase(token);
    token2block_.erase(token);
  }

  /*! \brief The mapping from each token to the binding block where it is created */
  std::unordered_map<const StorageToken*, const BindingBlockNode*> token2block_;
  /*! \brief The mapping from each token to the Exprs that share this token */
  std::unordered_map<const StorageToken*, std::vector<const ExprNode*>> token2exprs_;
};

class StorageAllocator : public StorageAllocatorBaseVisitor {
 public:
  explicit StorageAllocator(std::unordered_map<const ExprNode*, TokenContainer> token_map,
                            support::Arena* arena)
      : StorageAllocatorBaseVisitor(arena) {
    this->token_map_ = std::move(token_map);
  }

  std::string DumpMemoryAllocation() const { return allocator_.DumpMemoryAllocation(); }

  /*!
   * \brief The mapping from each memory-reusable `builtin.alloc_tensor` to its corresponding
   * underlying storage token that it is using.
   */
  std::unordered_map<const CallNode*, StorageToken*> alloc_tensor2token;
  /*! \brief The mapping from each Expr to the tensors that need to be killed after it. */
  std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors;
  /*! \brief The mapping from each binding block to the storage tokens that are create inside. */
  std::unordered_map<const BindingBlockNode*, std::unordered_set<const StorageToken*>> block2tokens;

 private:
  void VisitBindingBlock_(const BindingBlockNode* block) final {
    StorageAllocatorBaseVisitor::VisitBindingBlock_(block);
    // The algorithm guarantees that each the reference counter of storage token will be decreased
    // to 0 at the end of day. And we check the property here.
    for (const StorageToken* token : block2tokens[block]) {
      ICHECK_EQ(token->ref_counter, 0);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    StorageAllocatorBaseVisitor::VisitBinding_(binding);
    // If the binding is a memory-reusable `builtin.alloc_tensor`, map its underlying token to this
    // binding var, indicating that the token is currently occupied by this binding var.
    if (const CallNode* call_alloc_tensor = binding->value.as<CallNode>()) {
      auto it = alloc_tensor2token.find(call_alloc_tensor);
      if (it != alloc_tensor2token.end()) {
        auto it_insert = token2cur_tensor_.insert({it->second, binding->var});
        ICHECK(it_insert.second == true);
      }
    }
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op == alloc_tensor_op) {
      auto it = token_map_.find(call);
      ICHECK(it != token_map_.end());
      if (it->second.IsEmptyToken()) {
        return;
      }

      const auto* attrs = call->attrs.as<AllocTensorAttrs>();
      ICHECK_NOTNULL(attrs);
      ICHECK(!it->second.is_tuple);
      ICHECK(it->second.token != nullptr);
      StorageToken* new_token = this->RequestOrAlloc(it->second.token, attrs->runtime_device_index);
      // It doesn't make sense if a newly allocated tensor has 0 reference.
      ICHECK_GT(new_token->ref_counter, 0);

      alloc_tensor2token[call] = new_token;
      ExprUsesTokens(
          call, TokenContainer(/*is_tuple=*/false, /*field_containers=*/{}, /*token=*/new_token));
      ICHECK(!block_stack_.empty());
      block2tokens[block_stack_.back()].insert(new_token);
      return;
    }

    for (const Expr& arg : call->args) {
      const TokenContainer& container = GetTokens(arg);
      container.ApplyToTokens([this, call](StorageToken* token) {
        ICHECK_GT(token->ref_counter, 0);
        token->ref_counter -= 1;
        this->CheckForRelease(token, call);
      });
    }
  }

  StorageToken* RequestOrAlloc(StorageToken* prototype, int64_t virtual_device_idx) {
    StorageToken* token = allocator_.Request(prototype);
    if (token == nullptr) {
      token = allocator_.Alloc(prototype, this->n_storage_++);
    }
    ICHECK_NOTNULL(token);
    return token;
  }

  void CheckForRelease(StorageToken* token, const CallNode* release_site) {
    ICHECK_GE(token->storage_id, 0);
    ICHECK_GE(token->bytes, 0);
    ICHECK_GE(token->ref_counter, 0);
    ICHECK(!token->storage.defined());
    if (token->ref_counter == 0) {
      allocator_.Release(token);

      auto it = token2cur_tensor_.find(token);
      ICHECK(it != token2cur_tensor_.end());
      expr2killed_tensors[release_site].push_back(it->second);
      token2cur_tensor_.erase(it);
    }
  }

  /*! \brief Number of allocated storages */
  int n_storage_{0};
  /*! \brief The 1D memory allocator */
  TokenAllocator1D allocator_;
  /*! \brief The mapping from each token to the tensor that is currently occupying it */
  std::unordered_map<const StorageToken*, Var> token2cur_tensor_;
};

class StorageAllocationRewriter : public ExprMutator {
 public:
  explicit StorageAllocationRewriter(
      std::unordered_map<const CallNode*, StorageToken*> alloc_tensor2token,
      std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors,
      std::unordered_map<const BindingBlockNode*, std::unordered_set<const StorageToken*>>
          block2tokens,
      support::Arena* arena)
      : alloc_tensor2token_(std::move(alloc_tensor2token)),
        expr2killed_tensors_(std::move(expr2killed_tensors)),
        block2tokens_(std::move(block2tokens)),
        arena_(arena) {}

 private:
  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) final {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // Insert `memory.kill_storage` for the storage tokens allocated inside this block.
    for (const StorageToken* token : block2tokens_[block]) {
      ICHECK(token->storage.defined());
      this->builder_->Emit(MakeMemKillStorage(token->storage));
    }

    BindingBlock new_block = builder_->EndBlock();
    return new_block;
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    Expr new_value = this->builder_->Normalize(this->VisitExpr(binding->value));
    Var new_var = this->VisitVarDef(binding->var);
    ICHECK(!this->builder_->CurrentBlockIsDataFlow());

    if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
      this->builder_->Emit(GetRef<VarBinding>(binding));
    } else {
      Var temp = WithShapeAndType(new_var, new_value->shape_, new_value->checked_type_);
      if (!temp.same_as(new_var)) {
        new_var = temp;
        this->var_remap_[binding->var->vid] = new_var;
      }
      this->builder_->Emit(VarBinding(new_var, new_value));
    }

    // Insert `memory.kill_tensor` for the tensors that need to be killed after this binding.
    auto it = expr2killed_tensors_.find(binding->value.get());
    if (it != expr2killed_tensors_.end()) {
      for (const Var& var : it->second) {
        Var new_var = Downcast<Var>(this->VisitExpr(var));
        this->builder_->Emit(MakeMemKillTensor(new_var));
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto it = alloc_tensor2token_.find(call);
    if (it != alloc_tensor2token_.end()) {
      StorageToken* token = it->second;
      const auto* attrs = call->attrs.as<AllocTensorAttrs>();
      ICHECK_NOTNULL(attrs);
      // - If the token is visited for the first time, create a storage variable using
      // `memory.alloc_storage` for it.
      // - And always create a `memory.alloc_tensor` for the old `builtin.alloc_tensor`.
      if (!token->storage.defined()) {
        ShapeExpr size({tir::make_const(DataType::Int(64), token->bytes)});
        Call alloc_storage = Downcast<Call>(
            MakeAllocStorage(std::move(size), attrs->runtime_device_index, "global", token->dtype));
        token->storage = builder_->Emit(alloc_storage, "storage");
      }
      return MakeMemAllocTensor(token->storage, call->args[0], /*offset=*/0, attrs->dtype);
    }

    return ExprMutator::VisitExpr_(call);
  }

  /*!
   * \brief The mapping from each memory-reusable `builtin.alloc_tensor` to its corresponding
   * underlying storage token that it is using.
   */
  std::unordered_map<const CallNode*, StorageToken*> alloc_tensor2token_;
  /*! \brief The mapping from each Expr to the tensors that need to be killed after it. */
  std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors_;
  /*! \brief The mapping from each binding block to the storage tokens that are create inside. */
  std::unordered_map<const BindingBlockNode*, std::unordered_set<const StorageToken*>>
      block2tokens_;
  /*! \brief The allocator */
  support::Arena* arena_;
};

Expr VMGraphMemoryPlan(Function func) {
  support::Arena arena;
  // Step 1. Initialize.
  std::unordered_map<const ExprNode*, TokenContainer> token_map =
      StorageAllocatorInit(&arena).Initialize(func);
  // Step 2. Collect the memory allocation info.
  StorageAllocator allocator(std::move(token_map), &arena);
  allocator(func);
  // Dump the memory allocation information by using `allocator.DumpMemoryAllocation()`.
  // Step 3. Rewrite the function.
  StorageAllocationRewriter rewriter(std::move(allocator.alloc_tensor2token),
                                     std::move(allocator.expr2killed_tensors),
                                     std::move(allocator.block2tokens), &arena);
  func = Downcast<Function>(rewriter(func));
  return func;
}

namespace transform {

Pass VMGraphMemoryPlan() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(VMGraphMemoryPlan(std::move(f)));
      };
  return CreateFunctionPass(pass_func, 0, "VMGraphMemoryPlan", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMGraphMemoryPlan").set_body_typed(VMGraphMemoryPlan);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
