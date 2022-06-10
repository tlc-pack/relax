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

#include <algorithm>
#include <string>
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
 * \brief Bind the variables to a Relax expression. This is a helper
 * function usually called by other pass functions to help optimizations.
 * If any free variables are introduced into a function, those are added
 * to the function parameters.
 * Additionally this may change the order of parameters if you map a variable
 * to a variable.
 *
 * \param expr The input expression.
 * \param binds The variable to expression map that will be used to help the
 *        binding.
 *
 * \return The updated expression.
 */
TVM_DLL Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_UTILS_H_
