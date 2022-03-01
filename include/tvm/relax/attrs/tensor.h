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
 * \file tvm/relax/attrs/tensor.h
 * \brief Attributes for tensor allocation operators.
 */
#ifndef TVM_RELAX_ATTRS_TENSOR_H_
#define TVM_RELAX_ATTRS_TENSOR_H_

#include <tvm/ir/attrs.h>

namespace tvm {
namespace relax {
/*!
 * \brief Attributes for allocating tensors.
 */
struct AllocTensorAttrs : public tvm::AttrsNode<AllocTensorAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(AllocTensorAttrs, "relax.attrs.AllocTensorAttrs") {
    TVM_ATTR_FIELD(dtype).describe("The datatype of the tensor to be allocated.");
  }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ATTRS_TENSOR_H_
