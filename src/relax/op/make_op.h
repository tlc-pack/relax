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
 *
 * \file tvm/relax/op/make_op.h
 * \brief Header of internal operator functions
 * to assist in creating ops in C++
 */
#ifndef TVM_RELAX_OP_MAKE_OP_H_
#define TVM_RELAX_OP_MAKE_OP_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

Expr MakeCast(Expr data, DataType dtype);

Expr MakeAllocStorage(Expr size, int64_t virtual_device_index, std::string storage_scope,
                      DataType dtype);

Expr MakeMemAllocTensor(Expr storage, Expr shape, int offset, DataType dtype);

Expr MakeMemKillStorage(Expr storage);

Expr MakeMemKillTensor(Expr tensor);

Expr MakeVMAllocStorage(Expr size, DataType dtype, int64_t runtime_device_index);

Expr MakeVMAllocTensor(Expr storage, Expr shape, int offset, DataType dtype);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_MAKE_OP_H_
