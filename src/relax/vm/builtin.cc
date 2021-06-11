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
#include <tvm/runtime/memory.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace relax {
namespace vm {

using tvm::runtime::NDArray;

TVM_REGISTER_GLOBAL("vm.builtin.shape_of").set_body_typed(
[](NDArray arr) {
  return arr.Shape();
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_heap").set_body_typed(
[](int64_t size) {
  return NDArray::Empty(ShapeTuple({size}),
                        DLDataType{kDLInt, 64, 1},
                        DLDevice{kDLCPU, 0});
});

TVM_REGISTER_GLOBAL("vm.builtin.match_shape").set_body(
[] (runtime::TVMArgs args, runtime::TVMRetValue* rv) {
  ShapeTuple shape = args[0];
  NDArray heap = args[1];
  int64_t* heap_data = reinterpret_cast<int64_t*>(heap.ToDLPack()->dl_tensor.data);
  for (int i = 2; i < args.size(); ++i) {
    int64_t heap_idx = args[i];
    ICHECK(heap_idx >= 0 && heap_idx < heap.Shape()[0]);
    heap_data[heap_idx] = shape[i - 2];
  }
});

TVM_REGISTER_GLOBAL("vm.builtin.make_shape").set_body(
[](runtime::TVMArgs args, runtime::TVMRetValue* rv) {
  NDArray heap = args[0];
  int64_t* heap_data = reinterpret_cast<int64_t*>(heap.ToDLPack()->dl_tensor.data);
  std::vector<int64_t> shape;
  for (int i = 1; i < args.size(); ++i) {
    int64_t heap_idx = args[i];
    ICHECK(heap_idx >= 0 && heap_idx < heap.Shape()[0]);
    shape.push_back(heap_data[heap_idx]);
  }
  *rv = ShapeTuple(shape);
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_storage")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  void* vm_state_ptr = args[0];
  VMState* vm_state = static_cast<VMState*>(vm_state_ptr);
  Index size = args[1];
  Index alignment = args[2];
  Index device_type = args[3];
  DLDataType dtype_hint = args[4];
  DLOG(INFO) << "AllocStorage: allocation_size=" << size << ", alignment=" << alignment
              << ", dtype_hint=" << runtime::DLDataType2String(dtype_hint)
              << ", device_type=" << device_type;

  auto storage_obj = runtime::SimpleObjAllocator().make_object<StorageObj>();
  ICHECK_LT(static_cast<size_t>(device_type), vm_state->allocators.size())
      << "Memory allocator for device " << device_type << " has not been initialized";
  auto* alloc = vm_state->allocators[device_type];
  ICHECK(alloc) << "Did you forget to init the VirtualMachine with devices?";
  storage_obj->buffer = alloc->Alloc(size, alignment, dtype_hint);
  Storage storage(storage_obj);
  *rv = storage;
});

TVM_REGISTER_GLOBAL("vm.builtin.alloc_tensor")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Storage storage = args[0];
  Index offset = args[1];
  DLDataType dtype = args[2];
  ShapeTuple shape = args[3];
  auto obj = storage->AllocNDArray(offset, shape, dtype);
  *rv = obj;
});
}  // namespace vm
}  // namespace relax
}  // namespace tvm
