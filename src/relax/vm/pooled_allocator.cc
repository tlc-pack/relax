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

#include <tvm/relax/vm/pooled_allocator.h>

namespace tvm {
namespace relax {
namespace vm {

Buffer PooledAllocator::Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
  auto&& it = memory_pool_.find(size);
  if (it != memory_pool_.end() && !it->second.empty()) {
    auto&& pool = it->second;
    auto ret = pool.back();
    pool.pop_back();
    return ret;
  }
  Buffer buf;
  buf.device = device_;
  buf.size = size;
  try {
    buf.data =
        runtime::DeviceAPI::Get(device_)->AllocDataSpace(device_, size, alignment, type_hint);
  } catch (InternalError& err) {
    LOG(WARNING) << "PooledAllocator got InternalError during allocation: " << err.message();
    LOG(WARNING) << "Trying to release all unused memory and reallocate...";
    ReleaseAll();
    buf.data =
        runtime::DeviceAPI::Get(device_)->AllocDataSpace(device_, size, alignment, type_hint);
  }

  used_memory_.fetch_add(size, std::memory_order_relaxed);
  DLOG(INFO) << "allocate " << size << " B, used memory " << used_memory_ << " B";
  return buf;
}

void PooledAllocator::Free(const Buffer& buffer) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (memory_pool_.find(buffer.size) == memory_pool_.end()) {
    memory_pool_.emplace(buffer.size, std::vector<Buffer>{});
  }
  memory_pool_.at(buffer.size).push_back(buffer);
  DLOG(INFO) << "reclaim buffer " << buffer.size;
}

void PooledAllocator::ReleaseAll() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  for (auto const& it : memory_pool_) {
    auto const& pool = it.second;
    for (auto const& buf : pool) {
      runtime::DeviceAPI::Get(buf.device)->FreeDataSpace(buf.device, buf.data);
    }
  }
  memory_pool_.clear();
  used_memory_ = 0;
  DLOG(INFO) << "release all buffers";
}

}  // namespace vm
}  // namespace relax
}  // namespace tvm