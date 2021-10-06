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
 * \brief Options for allocating storage.
 */
struct AllocStorageAttrs : public tvm::AttrsNode<AllocStorageAttrs> {
  DataType dtype;
  int device_id;
  int device_type;

  TVM_DECLARE_ATTRS(AllocStorageAttrs, "relax.attrs.AllocStorageAttrs") {
    TVM_ATTR_FIELD(dtype)
        .describe("The dtype of the tensor to allocate.")
        .set_default(DataType::Float(32, 1));
    TVM_ATTR_FIELD(device_id).describe("The device id on which to allocate memory.");
    TVM_ATTR_FIELD(device_type).describe("The device type on which to allocate memory.");
  }
};

/*!
 * \brief Options for allocating tensors.
 */
struct AllocTensorAttrs : public tvm::AttrsNode<AllocTensorAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(AllocTensorAttrs, "relax.attrs.AllocTensorAttrs") {
    TVM_ATTR_FIELD(dtype)
        .describe("The dtype of the tensor to allocate.")
        .set_default(DataType::Float(32, 1));
  }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ATTRS_MEMORY_H_
