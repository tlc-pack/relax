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
 * \file tvm/relax/vm/transform.h
 * \brief Relax VM specific transformation passes.
 */
#ifndef TVM_RELAX_VM_TRANSFORM_H_
#define TVM_RELAX_VM_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
namespace relax {
namespace vm {
namespace transform {

using namespace tvm::relax::transform;
using Pass = tvm::transform::Pass;
using PassNode = tvm::transform::PassNode;
using PassInfo = tvm::transform::PassInfo;
using PassInfoNode = tvm::transform::PassInfoNode;
using PassContext = tvm::transform::PassContext;
using PassContextNode = tvm::transform::PassContextNode;
using Sequential = tvm::transform::Sequential;
using Function = tvm::relax::Function;

TVM_DLL Pass VMMemoryLower();

TVM_DLL Pass VMShapeLower();

}  // namespace transform
}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_VM_TRANSFORM_H_
