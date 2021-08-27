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
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/topi/broadcast.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

using Expr = tvm::RelayExpr;
using relay::Call;

#define RELAX_BINARY_COMPUTE(FTOPI)                       \
  [](const Attrs& attrs, const Array<te::Tensor>& inputs, \
     const Type& out_type) -> Array<te::Tensor> {         \
    ICHECK_EQ(inputs.size(), 2U);                         \
    return {FTOPI(inputs[0], inputs[1])};                 \
  }

RELAX_REGISTER_BINARY_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1);

RELAX_REGISTER_BINARY_OP("multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_support_level(1);

}  // namespace relax
}  // namespace tvm
