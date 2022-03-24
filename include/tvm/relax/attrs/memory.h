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
 * \file tvm/relax/attrs/memory.h
 * \brief Attributes for memory operators.
 */
#ifndef TVM_RELAX_ATTRS_MEMORY_H_
#define TVM_RELAX_ATTRS_MEMORY_H_

#include <tvm/ir/attrs.h>

namespace tvm {
namespace relax {

/*!
 * \brief Attributes for allocating tensor.
 */
struct AllocTensorAttrs : public tvm::AttrsNode<AllocTensorAttrs> {
  DataType dtype;
  int64_t runtime_device_index;

  TVM_DECLARE_ATTRS(AllocTensorAttrs, "relax.attrs.AllocTensorAttrs") {
    TVM_ATTR_FIELD(dtype).describe("The datatype of the tensor to be allocated.");
    TVM_ATTR_FIELD(runtime_device_index)
        .describe(
            "The device index indicating on which device the tensor is to be allocated at runtime. "
            "Index -1 is reserved for the host device.")
        .set_default(-1);
  }
};

/*!
 * \brief Attributes for allocating storage on Relax VM.
 */
struct VMAllocStorageAttrs : public tvm::AttrsNode<VMAllocStorageAttrs> {
  DataType dtype;
  int64_t runtime_device_index;

  TVM_DECLARE_ATTRS(VMAllocStorageAttrs, "relax.attrs.VMAllocStorageAttrs") {
    TVM_ATTR_FIELD(dtype)
        .describe("The dtype of the tensor to allocate.")
        .set_default(DataType::Float(32, 1));
    TVM_ATTR_FIELD(runtime_device_index)
        .describe(
            "The device index indicating on which device the tensor is to be allocated at runtime. "
            "Index -1 is reserved for the host device.")
        .set_default(-1);
  }
};

/*!
 * \brief Attributes for allocating tensor on Relax VM.
 */
struct VMAllocTensorAttrs : public tvm::AttrsNode<VMAllocTensorAttrs> {
  int offset;
  DataType dtype;

  TVM_DECLARE_ATTRS(VMAllocTensorAttrs, "relax.attrs.VMAllocTensorAttrs") {
    TVM_ATTR_FIELD(offset).describe("Storage offset to allocate the tensor.").set_default(0);
    TVM_ATTR_FIELD(dtype)
        .describe("The dtype of the tensor to allocate.")
        .set_default(DataType::Float(32, 1));
  }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ATTRS_MEMORY_H_
