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
 * \file src/relax/vm/builtin.cc
 * \brief
 */
#include <tvm/relax/vm/bytecode.h>
#include <tvm/relax/vm/memory_manager.h>
#include <tvm/relax/vm/vm.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

TVM_REGISTER_GLOBAL("vm.builtin.shape_of")
.set_body_typed([](NDArray arr) {
  return arr.Shape();
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_shape_heap")
.set_body_typed([](ShapeTuple size) {
  return NDArray::Empty(size, DLDataType{kDLInt, 64, 1}, DLDevice{kDLCPU, 0});
});

TVM_REGISTER_GLOBAL("vm.builtin.store_shape")
.set_body_typed([](ShapeTuple shape, NDArray heap, ShapeTuple indexes) {
  int64_t* heap_data = reinterpret_cast<int64_t*>(heap.ToDLPack()->dl_tensor.data);
  for (size_t i = 0; i < indexes.size(); ++i) {
    int64_t heap_idx = indexes[i];
    ICHECK(heap_idx >= 0 && heap_idx < heap.Shape()[0]);
    heap_data[heap_idx] = shape[i];
  }
});

TVM_REGISTER_GLOBAL("vm.builtin.load_shape")
.set_body_typed([](NDArray heap, ShapeTuple indexes) {
  int64_t* heap_data = reinterpret_cast<int64_t*>(heap.ToDLPack()->dl_tensor.data);
  std::vector<int64_t> shape;
  for (size_t i = 0; i < indexes.size(); ++i) {
    int64_t heap_idx = indexes[i];
    ICHECK(heap_idx >= 0 && heap_idx < heap.Shape()[0]);
    shape.push_back(heap_data[heap_idx]);
  }
  return ShapeTuple(shape);
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_storage")
.set_body_typed([](void* vm_state_ptr, ShapeTuple buffer_size, Index device_type, DLDataType dtype_hint) {
  int alignment = runtime::kAllocAlignment;
  ICHECK_EQ(buffer_size.size(), 1);
  VMState* vm_state = static_cast<VMState*>(vm_state_ptr);
  int64_t size_imm = buffer_size[0];
  DLOG(INFO) << "AllocStorage: allocation_size=" << size_imm << ", alignment=" << alignment
              << ", dtype_hint=" << runtime::DLDataType2String(dtype_hint)
              << ", device_type=" << device_type;

  auto storage_obj = runtime::SimpleObjAllocator().make_object<StorageObj>();
  ICHECK_LT(static_cast<size_t>(device_type), vm_state->allocators.size())
      << "Memory allocator for device " << device_type << " has not been initialized";
  auto* alloc = vm_state->allocators[device_type];
  ICHECK(alloc) << "Did you forget to init the VirtualMachine with devices?";
  storage_obj->buffer = alloc->Alloc(size_imm, alignment, dtype_hint);
  Storage storage(storage_obj);
  return storage;
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_tensor")
.set_body_typed([](Storage storage, ShapeTuple shape, Index offset, DLDataType dtype) {
  auto tensor = storage->AllocNDArray(offset, shape, dtype);
  return tensor;
});

TVM_REGISTER_GLOBAL("vm.binary_broadcast_shape_infer")
.set_body_typed([](ShapeTuple lhs_shape, ShapeTuple rhs_shape) {
  std::vector<int64_t> output_shape;
  size_t ndim0 = lhs_shape.size();
  size_t ndim1 = rhs_shape.size();
  size_t i = 1;
  for (; i <= std::min(ndim0, ndim1); ++i) {
    int64_t lhs_dim = lhs_shape[ndim0 - i];
    int64_t rhs_dim = rhs_shape[ndim1 - i];
    if (lhs_dim == 1 || rhs_dim == 1 || lhs_dim == rhs_dim) {
      output_shape.push_back(std::max(lhs_dim, rhs_dim));
    } else {
      LOG(FATAL) << "Incompatible shapes " << lhs_shape << " and " << rhs_shape
                  << " for broadcasting";
    }
  }
  size_t max_ndim = std::max(ndim0, ndim1);
  ShapeTuple& longer_shape = (ndim0 > ndim1) ? lhs_shape : rhs_shape;
  for (; i <= max_ndim; ++i) {
    output_shape.push_back(longer_shape[max_ndim - i]);
  }
  return ShapeTuple(output_shape.rbegin(), output_shape.rend());
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
