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
 * \file tvm/relax/attrs/builtin.h
 * \brief Attributes for relax low-level builtin operators.
 */
#ifndef TVM_RELAX_ATTRS_BUILTIN_H_
#define TVM_RELAX_ATTRS_BUILTIN_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relax {

/*!
 * \brief Common attribute for built-in runtime packed functions.
 */
struct BuiltinFuncAttrs : public tvm::AttrsNode<BuiltinFuncAttrs> {
  Array<IntImm> int_args;
  DataType dtype_arg;
  Array<String> str_args;
  bool require_ctx;

  TVM_DECLARE_ATTRS(BuiltinFuncAttrs, "relax.attrs.BuiltinPackedAttrs") {
    TVM_ATTR_FIELD(int_args)
        .describe("List of additional int arguments to pass to the function.")
        .set_default(NullValue<Array<IntImm>>());
    TVM_ATTR_FIELD(dtype_arg)
        .describe("Optional dtype argument to pass to the function.")
        .set_default(NullValue<DataType>());
    TVM_ATTR_FIELD(str_args)
        .describe("List of additional string arguments to pass to the function.")
        .set_default(NullValue<Array<String>>());
    TVM_ATTR_FIELD(require_ctx)
        .describe("Where we need to pass in ctx ptr as first argument.")
        .set_default(false);
  }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ATTRS_BUILTIN_H_
