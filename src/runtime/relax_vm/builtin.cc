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
 * \file src/runtime/relax_vm/builtin.cc
 */
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/runtime/relax_vm/memory_manager.h>
#include <tvm/runtime/relax_vm/vm.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

TVM_REGISTER_GLOBAL("vm.builtin.shape_of").set_body_method(&NDArray::Shape);

TVM_REGISTER_GLOBAL("vm.builtin.copy").set_body_typed([](NDArray src) { return src; });

TVM_REGISTER_GLOBAL("vm.builtin.alloc_shape_heap")
    .set_body_typed([](void* vm_ptr, ShapeTuple size) {
      VirtualMachine* vm = static_cast<VirtualMachine*>(vm_ptr);
      return NDArray::Empty(size, DLDataType{kDLInt, 64, 1}, vm->devices[0]);
    });

TVM_REGISTER_GLOBAL("vm.builtin.alloc_closure").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<ObjectRef> cap_vars;
  for (int i = 1; i < args.size(); ++i) {
    cap_vars.push_back(args[i]);
  }
  String func_name = args[0];
  VMClosure vm_closure(func_name, cap_vars);

  *rv = std::move(vm_closure);
});

TVM_REGISTER_GLOBAL("vm.builtin.invoke_closure").set_body([](TVMArgs args, TVMRetValue* rv) {
  // args[0]: vm; args[1]: closure; args[2, 3, ...]: function arguments
  void* vm_ptr = args[0];
  VirtualMachine* vm = static_cast<VirtualMachine*>(vm_ptr);
  VMClosure vm_closure = args[1];
  runtime::String func_name = vm_closure->func_name;

  PackedFunc func{nullptr};
  func = vm->GetFunction(func_name, GetObjectPtr<Object>(vm));
  ICHECK(func != nullptr) << "cannot find closure " << func_name;

  // get closure free_vars
  Array<ObjectRef> cap_vars = vm_closure->free_vars;
  size_t num_tensor_args = args.size() - 2;
  std::vector<TVMValue> values(num_tensor_args + cap_vars.size());
  std::vector<int> tcodes(num_tensor_args + cap_vars.size());

  runtime::TVMArgsSetter setter(values.data(), tcodes.data());
  for (size_t i = 0; i < num_tensor_args; i++) {
    NDArray arg = args[i + 2];
    setter(i, arg);
  }
  for (size_t i = 0; i < cap_vars.size(); i++) {
    setter(i + num_tensor_args, cap_vars[i]);
  }
  TVMArgs func_args(values.data(), tcodes.data(), values.size());
  func.CallPacked(func_args, rv);
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

TVM_REGISTER_GLOBAL("vm.builtin.load_shape").set_body_typed([](NDArray heap, ShapeTuple indexes) {
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
    .set_body_typed([](void* vm_ptr, ShapeTuple buffer_size, Index device_index,
                       DLDataType dtype_hint) {
      ICHECK_EQ(buffer_size.size(), 1);
      int alignment = runtime::kAllocAlignment;
      VirtualMachine* vm = static_cast<VirtualMachine*>(vm_ptr);
      ICHECK_LT(device_index, vm->devices.size())
          << "The device index is out of VM physical devices list";

      if (device_index == -1) {
        // Allocate on host. Host is always the last element of vm->devices.
        device_index = vm->devices.size() - 1;
      }

      int64_t size_imm = buffer_size[0];

      auto storage_obj = runtime::SimpleObjAllocator().make_object<StorageObj>();
      auto* alloc = vm->allocators[device_index];
      ICHECK(alloc) << "Did you forget to init the VirtualMachine with devices?";
      storage_obj->buffer = alloc->Alloc(size_imm, alignment, dtype_hint);
      Storage storage(storage_obj);
      return storage;
    });

TVM_REGISTER_GLOBAL("vm.builtin.alloc_tensor").set_body_method<Storage>(&StorageObj::AllocNDArray);

TVM_REGISTER_GLOBAL("vm.binary_broadcast_shape_infer")
    .set_body_typed([](ShapeTuple lhs_shape, ShapeTuple rhs_shape) {
      std::vector<int64_t> output_shape;
      size_t ndim0 = lhs_shape.size();
      size_t ndim1 = rhs_shape.size();
      size_t i = 1;
      for (; i <= std::min(ndim0, ndim1); ++i) {
        int64_t lhs_dim = lhs_shape[ndim0 - i];
        int64_t rhs_dim = rhs_shape[ndim1 - i];
        ICHECK(lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1);
        output_shape.push_back(std::max(lhs_dim, rhs_dim));
      }
      size_t max_ndim = std::max(ndim0, ndim1);
      ShapeTuple& longer_shape = (ndim0 > ndim1) ? lhs_shape : rhs_shape;
      for (; i <= max_ndim; ++i) {
        output_shape.push_back(longer_shape[max_ndim - i]);
      }
      return ShapeTuple(output_shape.rbegin(), output_shape.rend());
    });

TVM_REGISTER_GLOBAL("vm.call_tir_dyn").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* vm_ptr = args[0];
  VirtualMachine* vm = static_cast<VirtualMachine*>(vm_ptr);
  runtime::String func_name = args[1];

  PackedFunc func{nullptr};
  if (vm->lib.defined()) {
    func = vm->lib.value()->GetFunction(func_name, true);
  }
  if (!func.defined()) {
    const PackedFunc* p_func = Registry::Get(func_name);
    CHECK(p_func != nullptr);
    func = *(p_func);
  }

  ShapeTuple to_unpack = args[args.size() - 1];
  size_t num_tensor_args = args.size() - 3;
  std::vector<TVMValue> values(num_tensor_args + to_unpack.size());
  std::vector<int> tcodes(num_tensor_args + to_unpack.size());
  runtime::TVMArgsSetter setter(values.data(), tcodes.data());
  for (size_t i = 0; i < num_tensor_args; i++) {
    NDArray arg = args[i + 2];
    setter(i, arg);
  }
  for (size_t i = 0; i < to_unpack.size(); i++) {
    setter(i + num_tensor_args, to_unpack[i]);
  }

  TVMArgs func_args(values.data(), tcodes.data(), values.size());
  func.CallPacked(func_args, rv);
});

TVM_REGISTER_GLOBAL("vm.runtime.TupleGetItem")
    .set_body_typed([](runtime::ADT adt, ShapeTuple index) {
      ICHECK_EQ(index.size(), 1);
      int idx = index[0];
      ICHECK_LT(idx, adt.size());
      return adt[idx];
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
