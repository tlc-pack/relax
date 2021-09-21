/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file memory.cc
 * \brief memory related operators.
 */

#include "memory.h"

namespace tvm {
namespace relax {

// alloc_storage

RELAY_REGISTER_OP("relax.alloc_storage")
.set_num_inputs(1)
.add_argument("size", "Expr", "The size of the storage to allocate.")
.set_attr<FInferShape>("FInferShape", InferShapeAllocStorage)
.set_attr<FInferType>("FInferType", InferTypeAllocStorage);

Expr MakeAllocStorage(Expr size) {
  static const Op& op = Op::Get("relax.alloc_storage");
  return Call(op, {size}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.alloc_storage")
.set_body_typed(MakeAllocStorage);

// alloc_tensor

RELAY_REGISTER_OP("relax.alloc_tensor")
.set_num_inputs(1)
.add_argument("storage", "Var", "The storage to allocate from.")
.add_argument("offset", "Expr", "The offset into the backing storage.")
.add_argument("shape", "Expr", "The shape of the tensor to allocate.")
.set_attr<FInferShape>("FInferShape", InferShapeAllocTensor)
.set_attr<FInferType>("FInferType", InferTypeAllocTensor);

Expr MakeAllocTensor(Var storage, Expr offset, Expr shape) {
  static const Op& op = Op::Get("relax.alloc_tensor");
  return Call(op, {storage, offset, shape}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.alloc_tensor")
.set_body_typed(MakeAllocTensor);

}  // namespace relax
}  // namespace tvm
