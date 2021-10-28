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
 * \file tvm/relax/utils.h
 * \brief Utility classes and functions for working with the Relax IR.
 */
#ifndef TVM_RELAX_UTILS_H_
#define TVM_RELAX_UTILS_H_

#include <tvm/relax/expr.h>

#include <string>
#include <algorithm>
#include <unordered_map>

namespace tvm {
namespace relax {

/*!
 * \brief Utility data structure for generating unique names for IR construction.
 */
class NameTable {
 public:
  /*!
   * \brief Generate a unique name with a specified prefix.
   * \param prefix The name prefix.
   * \return The generated name.
   */
  inline std::string GetUniqueName(std::string prefix) {
    std::replace(prefix.begin(), prefix.end(), '.', '_');
    std::string unique_prefix = prefix;
    auto it = alloc_map_.find(prefix);
    if (it != alloc_map_.end()) {
      while (alloc_map_.count(unique_prefix = prefix + std::to_string(++it->second)) > 0) {
      }
    }
    alloc_map_[unique_prefix] = 0;
    return unique_prefix;
  }

 private:
  std::unordered_map<std::string, uint32_t> alloc_map_;
};

/*!
 * \brief Memoization map for expressions using Id for equality of variables.
 */
class ExprMemo {
 public:
  Optional<Expr> Get(const Expr& expr) {
    if (const VarNode* var = expr.as<VarNode>()) {
      auto it = var_memo_.find(var->vid);
      if (it != var_memo_.end()) {
        return it->second;
      }
    } else {
      auto it = expr_memo_.find(expr);
      if (it != expr_memo_.end()) {
        return it->second;
      }
    }
    return NullOpt;
  }

  void Set(const Expr& pre, const Expr& post) {
    if (const VarNode* var = pre.as<VarNode>()) {
      var_memo_[var->vid] = post;
    } else {
      expr_memo_[pre] = post;
    }
  }

 private:
  std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual> var_memo_;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> expr_memo_;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_UTILS_H_
