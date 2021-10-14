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
 * \file tvm/relax/name_table.h
 * \brief Utility data structure for generating unique names for IR construction.
 */
#ifndef TVM_RELAX_NAME_TABLE_H_
#define TVM_RELAX_NAME_TABLE_H_

#include <algorithm>
#include <string>
#include <unordered_map>

namespace tvm {
namespace relax {

// TODO(@altanh): FFI?
class NameTable {
 public:
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

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_NAME_TABLE_H_