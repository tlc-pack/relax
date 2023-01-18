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
 * \file unary.cc
 * \brief Relax unary arithmetic operators.
 */

#include "unary.h"

namespace tvm {
namespace relax {

RELAX_REGISTER_UNARY_OP_INTERFACE(cos, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_OP_INTERFACE(exp, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_OP_INTERFACE(log, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_OP_INTERFACE(negative, /*require_float_dtype=*/false);
RELAX_REGISTER_UNARY_OP_INTERFACE(sigmoid, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_OP_INTERFACE(sin, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_OP_INTERFACE(sqrt, /*require_float_dtype=*/true);
RELAX_REGISTER_UNARY_OP_INTERFACE(tanh, /*require_float_dtype=*/true);

}  // namespace relax
}  // namespace tvm
