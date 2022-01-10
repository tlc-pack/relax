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

#include <tvm/relax/topi.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("relax.mean").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::mean(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("relax.variance").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::variance(args[0], args[1], args[2], args[3], args[4], args[5]);
});

TVM_REGISTER_GLOBAL("relax.reshape").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::reshape(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("relax.reverse_reshape").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::reshape(args[0], args[1], true);
});

TVM_REGISTER_GLOBAL("relax.bias_add").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::bias_add(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("relax.collapse_sum").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::collapse_sum(args[0], args[1]);
});

}  // namespace topi
}  // namespace relax
}  // namespace tvm
