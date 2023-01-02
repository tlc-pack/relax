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
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/builtin.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/runtime/relax_vm/memory_manager.h>
#include <tvm/runtime/relax_vm/vm.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

//-------------------------------------------------
//  Shape/StructInfo handling.
//-------------------------------------------------
/*!
 * \brief Builtin function to allocate shape heap.
 * \param ctx_ptr The context module pointer.
 * \param size the size of the heap.
 * \return An allocate NDArray as shape heap.
 */
NDArray AllocShapeHeap(void* ctx_ptr, int64_t size) {
  VirtualMachine* vm = static_cast<VirtualMachine*>(ctx_ptr);
  // use host allocator, which is always last element.
  size_t host_device_index = vm->devices.size() - 1;
  // specialy handle hexagon on-device RT.
  // TODO(relax-team): visit and consider other possible choices.
  if (vm->devices[0].device_type == kDLHexagon) {
    host_device_index = 0;
  }
  auto* alloc = vm->allocators[host_device_index];
  return alloc->Empty({size}, DLDataType{kDLInt, 64, 1}, vm->devices[host_device_index]);
}

TVM_REGISTER_GLOBAL("vm.builtin.alloc_shape_heap").set_body_typed(AllocShapeHeap);

/*!
 * \brief Builtin match shape function.
 * \param args The packed function arguments.
 * \param rv The return value.
 *
 * \sa MatchShapeCode
 */
void MatchShape(TVMArgs args, TVMRetValue* rv) {
  // input shape the first argument can take in tensor or shape.
  ShapeTuple input_shape;
  if (args[0].IsObjectRef<NDArray>()) {
    input_shape = args[0].operator NDArray().Shape();
  } else {
    input_shape = args[0];
  }
  DLTensor* heap = args[1];
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);
  int64_t size = args[2];
  const int64_t kBeginCode = 3;
  ICHECK_LE(kBeginCode + size * 2, args.size());
  // a function that lazily get context for error reporting
  const int64_t kErrorContextOffset = kBeginCode + size * 2;
  Optional<String> err_ctx = args[kErrorContextOffset];

  CHECK_EQ(input_shape.size(), size)
      << "RuntimeError: " << err_ctx.value_or("") << " match_cast shape size mismatch.";

  for (int64_t i = 0; i < size; ++i) {
    MatchShapeCode code = static_cast<MatchShapeCode>(args[kBeginCode + i * 2].operator int());
    int64_t reg = args[kBeginCode + i * 2 + 1];

    if (code == MatchShapeCode::kAssertEqualToImm) {
      CHECK_EQ(input_shape[i], reg)
          << "RuntimeError: " << err_ctx.value_or("") << " match_cast error, "
          << " shape[" << i << "]"
          << " mismatch to specified constant.";
    } else if (code == MatchShapeCode::kStoreToHeap) {
      heap_data[reg] = input_shape[i];
    } else if (code == MatchShapeCode::kNoOp) {
    } else {
      ICHECK(code == MatchShapeCode::kAssertEqualToLoad);
      CHECK_EQ(input_shape[i], heap_data[reg])
          << "RuntimeError: " << err_ctx.value_or("") << " match_cast error, "
          << " shape[" << i << "]"
          << " mismatch to a previous populated value.";
    }
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.match_shape").set_body(MatchShape);

/*!
 * \brief Builtin make shape function.
 * \param args The packed function arguments.
 * \param rv The return value.
 *
 * \sa MakeShapeCode
 */
void MakeShape(TVMArgs args, TVMRetValue* rv) {
  // NOTE: heap can be nullptr
  DLTensor* heap = args[0];
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);
  int64_t size = args[1];
  const int64_t kBeginCode = 2;

  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; ++i) {
    MakeShapeCode code = static_cast<MakeShapeCode>(args[kBeginCode + i * 2].operator int());
    int64_t reg = args[kBeginCode + i * 2 + 1];
    if (code == MakeShapeCode::kUseImm) {
      shape[i] = reg;
    } else {
      ICHECK(code == MakeShapeCode::kLoadShape);
      shape[i] = heap_data[reg];
    }
  }
  *rv = ShapeTuple(std::move(shape));
}

TVM_REGISTER_GLOBAL("vm.builtin.make_shape").set_body(MakeShape);

/*!
 * \brief Builtin function to check if arg is Tensor(dtype, ndim)
 * \param arg The input argument.
 * \param ndim Expected ndim of the Tensor, can be -1 (indicate unknown).
 * \paramf dtype The expected content data type.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTensorInfo(TVMArgs args, TVMRetValue* rv) {
  ObjectRef arg = args[0];
  int ndim = args[1];
  DataType dtype;
  Optional<String> err_ctx;

  if (args.size() == 3) {
    dtype = DataType::Void();
    err_ctx = args[2].operator Optional<String>();
  } else {
    dtype = args[2];
    err_ctx = args[3].operator Optional<String>();
  }

  auto* ptr = arg.as<NDArray::ContainerType>();
  CHECK(ptr != nullptr) << "TypeError: " << err_ctx.value_or("") << " expect a Tensor but get "
                        << arg->GetTypeKey();

  if (ndim != -1) {
    CHECK(ptr->dl_tensor.ndim == ndim)
        << "ValueError: " << err_ctx.value_or("") << " expect Tensor with ndim " << ndim
        << " but get " << ptr->dl_tensor.ndim;
  }

  if (dtype != DataType::Void()) {
    CHECK(DataType(ptr->dl_tensor.dtype) == dtype)
        << "ValueError: " << err_ctx.value_or("") << " expect Tensor with dtype " << dtype
        << " but get " << ptr->dl_tensor.dtype;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.check_tensor_info").set_body(CheckTensorInfo);

/*!
 * \brief Builtin function to check if arg is Shape(ndim)
 * \param arg The input argument.
 * \param ndim Expected size of the shape, can be -1 (indicate unknown).
 * \param err_ctx Additional context if error occurs.
 */
void CheckShapeInfo(ObjectRef arg, int ndim, Optional<String> err_ctx) {
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<ShapeTuple::ContainerType>();
  CHECK(ptr != nullptr) << "TypeError: " << err_ctx.value_or("") << " expect a Shape but get "
                        << arg->GetTypeKey();
  if (ndim != -1) {
    CHECK(ptr->size == static_cast<uint64_t>(ndim))
        << "ValueError: " << err_ctx.value_or("") << " expect Shape with ndim " << ndim
        << " but get " << ptr->size;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.check_shape_info").set_body_typed(CheckShapeInfo);

/*!
 * \brief Builtin function to check if arg is Tuple with size elements.
 * \param arg The input argument.
 * \param size The expected size of the tuple.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTupleInfo(ObjectRef arg, int64_t size, Optional<String> err_ctx) {
  using Tuple = runtime::ADT;
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<Tuple::ContainerType>();
  CHECK(ptr != nullptr) << "TypeError: " << err_ctx.value_or("") << " expect a Tuple but get "
                        << arg->GetTypeKey();
  CHECK(static_cast<int64_t>(ptr->size) == size)
      << "ValueError: " << err_ctx.value_or("") << " expect a Tuple with " << size << " elements, "
      << " but get a Tuple with " << ptr->size << " elements.";
}

TVM_REGISTER_GLOBAL("vm.builtin.check_tuple_info").set_body_typed(CheckTupleInfo);

/*!
 * \brief Builtin function to check if arg is a callable function.
 * \param arg The input argument.
 * \param err_ctx Additional context if error occurs.
 */
void CheckFuncInfo(ObjectRef arg, Optional<String> err_ctx) {
  // a function that lazily get context for error reporting
  bool is_func = arg.as<PackedFunc::ContainerType>() || arg.as<VMClosure::ContainerType>();
  CHECK(is_func) << "TypeError: " << err_ctx.value_or("") << " expect a Function but get "
                 << arg->GetTypeKey();
}

TVM_REGISTER_GLOBAL("vm.builtin.check_func_info").set_body_typed(CheckFuncInfo);

//-------------------------------------------------
//  Storage management.
//-------------------------------------------------
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

//-------------------------------------------------
//  Closure function handling, calling convention
//-------------------------------------------------
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

TVM_REGISTER_GLOBAL("vm.call_tir_dyn").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* vm_ptr = args[0];
  VirtualMachine* vm = static_cast<VirtualMachine*>(vm_ptr);
  // TODO(relax-team): directly pass in func instead of func name.
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

//-------------------------------------
//  Builtin runtime operators.
//-------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.shape_of").set_body_method(&NDArray::Shape);

TVM_REGISTER_GLOBAL("vm.builtin.copy").set_body_typed([](NDArray src) { return src; });

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

//-------------------------------------
//  Data structure API
//-------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.tuple_getitem").set_body_typed([](runtime::ADT adt, int64_t index) {
  return adt[index];
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
