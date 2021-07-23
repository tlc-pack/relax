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
 * \file src/relax/vm/naive_allocator.cc
 * \brief
 */

#include <tvm/relax/vm/naive_allocator.h>

namespace tvm {
namespace relax {
namespace vm {

Buffer NaiveAllocator::Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) {
  Buffer buf;
  buf.device = device_;
  buf.size = nbytes;
  buf.data =
      runtime::DeviceAPI::Get(device_)->AllocDataSpace(device_, nbytes, alignment, type_hint);
  used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
  DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
  return buf;
}

void NaiveAllocator::Free(const Buffer& buffer) {
  runtime::DeviceAPI::Get(device_)->FreeDataSpace(buffer.device, buffer.data);
  used_memory_.fetch_sub(buffer.size, std::memory_order_relaxed);
  DLOG(INFO) << "free " << buffer.size << " B, used memory " << used_memory_ << " B";
}

}  // namespace vm
}  // namespace relax
}  // namespace tvm