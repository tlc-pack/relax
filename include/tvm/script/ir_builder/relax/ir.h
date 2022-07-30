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
#ifndef TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_
#define TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_

#include <tvm/relax/expr.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/relax/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

////////////////////////////// Tensor Type //////////////////////////////

tvm::relax::Var Tensor(Optional<Array<PrimExpr>> shape, DataType dtype,
                       Optional<Integer> ndim = NullOpt);

/////////////////////////////// Function ////////////////////////////////

FunctionFrame Function();
tvm::relax::Var Arg(const String& name, const tvm::relax::Var& var);
void FuncName(const String& name);
void FuncAttrs(Map<String, ObjectRef> attrs);
tvm::Type RetType(tvm::Type ret_type);
void FuncReturn(const tvm::relax::Expr& value);

///////////////////////////// BindingBlock //////////////////////////////

BlockFrame BindingBlock();
BlockFrame Dataflow();

////////////////////////////// Bindings ////////////////////////////////

tvm::relax::Var Emit(const tvm::relax::Expr& expr);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_IR_IR_BUILDER_IR_IR_H_
