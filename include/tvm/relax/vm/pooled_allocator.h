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
 * \file tvm/relax/vm/pooled_allocator.h
 */
#ifndef TVM_RELAX_VM_POOLED_ALLOCATOR_H_
#define TVM_RELAX_VM_POOLED_ALLOCATOR_H_

#include <tvm/relax/vm/memory_manager.h>
#include <tvm/runtime/device_api.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {
namespace vm {

class PooledAllocator final : public Allocator {
 public:
  static constexpr size_t kDefaultPageSize = 4096;

  explicit PooledAllocator(Device dev, size_t page_size = kDefaultPageSize)
      : Allocator(kPooled), page_size_(page_size), used_memory_(0), device_(dev) {}

  ~PooledAllocator() { ReleaseAll(); }

  Buffer Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) override;

  void Free(const Buffer& buffer) override;

 private:
  void ReleaseAll();

 private:
  size_t page_size_;
  std::atomic<size_t> used_memory_;
  std::unordered_map<size_t, std::vector<Buffer> > memory_pool_;
  std::recursive_mutex mu_;
  Device device_;
};

}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_POOLED_ALLOCATOR_H_
