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
#ifndef TVM_SCRIPT_IR_BUILDER_TIR_IR_H_
#define TVM_SCRIPT_IR_BUILDER_TIR_IR_H_

#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/tir/frame.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

using tvm::runtime::NDArray;
using tvm::tir::Buffer;
using tvm::tir::Var;

Buffer BufferDecl(Array<PrimExpr> shape, DataType dtype, String buffer_name, Optional<Var> data,
                  Optional<Array<PrimExpr>> strides, Optional<PrimExpr> elem_offset,
                  String storage_scope, int align, int offset_factor, String buffer_type,
                  Optional<Array<IntImm>> axis_separators);
PrimExpr Ptr(runtime::DataType dtype, String storage_scope = "global");

BlockFrame Block(String name, bool no_realize = false);
BlockInitFrame Init();
void Where(PrimExpr predicate);
void Reads(Array<ObjectRef> buffer_slices);
void Writes(Array<ObjectRef> buffer_slices);
void BlockAttrs(Map<String, ObjectRef> attrs);
Buffer AllocBuffer(Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                   Optional<Var> data = NullOpt, Array<PrimExpr> strides = {},
                   PrimExpr elem_offset = PrimExpr(), String storage_scope = "", int align = -1,
                   int offset_factor = 0, String buffer_type = "default",
                   Array<IntImm> axis_separators = {});

namespace axis {
Var Spatial(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));
Var Reduce(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));
Var Scan(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));
Var Opaque(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));
Array<Var> Remap(String kinds, Array<PrimExpr> bindings, DataType dtype = DataType::Int(32));
}  // namespace axis

ForFrame Serial(PrimExpr start, PrimExpr stop,
                Optional<Map<String, ObjectRef>> annotations = NullOpt);
ForFrame Parallel(PrimExpr start, PrimExpr stop,
                  Optional<Map<String, ObjectRef>> annotations = NullOpt);
ForFrame Vectorized(PrimExpr start, PrimExpr stop,
                    Optional<Map<String, ObjectRef>> annotations = NullOpt);
ForFrame Unroll(PrimExpr start, PrimExpr stop,
                Optional<Map<String, ObjectRef>> annotations = NullOpt);
ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, String thread,
                       Optional<Map<String, ObjectRef>> annotations = NullOpt);
ForFrame Grid(Array<PrimExpr> extents);

PrimFuncFrame PrimFunc();
Var Arg(String name, Var var);
Buffer Arg(String name, Buffer buffer);
void FuncName(String name);
void FuncAttrs(Map<String, ObjectRef> attrs);
Type FuncRet(Type ret_type);
Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                   Optional<Var> data = NullOpt, Array<PrimExpr> strides = {},
                   PrimExpr elem_offset = PrimExpr(), String storage_scope = "global",
                   int align = -1, int offset_factor = 0, String buffer_type = "default",
                   Array<IntImm> axis_separators = {});
void PreflattenedBuffer(Buffer postflattened_buffer, Array<PrimExpr> shape,
                        DataType dtype = DataType::Float(32), Optional<Var> data = NullOpt,
                        Array<PrimExpr> strides = {}, PrimExpr elem_offset = PrimExpr(),
                        String storage_scope = "global", int align = -1, int offset_factor = 0,
                        String buffer_type = "default", Array<IntImm> axis_separators = {});

/*!
 * \brief The assertion statement.
 * \param condition The assertion condition.
 * \param message The error message when the assertion fails.
 * \return The AssertFrame.
 */
AssertFrame Assert(PrimExpr condition, String message);
/*!
 * \brief The let binding.
 * \param var The variable to bind.
 * \param value The value to be bound.
 * \return The created LetFrame.
 */
LetFrame Let(Var var, PrimExpr value);
AllocateFrame Allocate(Array<PrimExpr> extents, DataType dtype, String storage_scope = "",
                       Optional<PrimExpr> condition = NullOpt,
                       Optional<Map<String, ObjectRef>> annotations = NullOpt);
AllocateConstFrame AllocateConst(
    NDArray data, DataType dtype, Array<PrimExpr> extents,
    Map<String, ObjectRef> annotations = NullValue<Map<String, ObjectRef>>());
/*!
 * \brief The realization.
 * \param buffer_slice The region of buffer access.
 * \param storage_scope The storage scope associated with this realization.
 * \param condition The condition expression.
 * \return The result RealizeFrame.
 */
RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope, PrimExpr condition);
AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value);
WhileFrame While(PrimExpr condition);
IfFrame If(PrimExpr condition);
ThenFrame Then();
ElseFrame Else();
/*!
 * \brief Launch a thread.
 * \param var The iteration variable.
 * \param extent The extent of environment thread.
 * \return The result LaunchThreadFrame.
 */
LaunchThreadFrame LaunchThread(Var var, PrimExpr extent);
/*!
 * \brief Bind a var to thread env.
 * \param thread_tag The thread type tag.
 * \return The result variable which gets bound to the thread env.
 */
Var EnvThread(String thread_tag);
void BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices);
void Prefetch(Buffer buffer, Array<Range> bounds);
/*!
 * \brief Evaluate the input expression.
 * \param value The input expression to evaluate.
 */
void Evaluate(PrimExpr value);

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName, DType)                             \
  inline PrimExpr FuncName(Optional<PrimExpr> expr = NullOpt) {                        \
    DataType dtype = DType;                                                            \
    return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype); \
  }

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int8, DataType::Int(8));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int16, DataType::Int(16));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int32, DataType::Int(32));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int64, DataType::Int(64));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(UInt8, DataType::UInt(8));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(UInt16, DataType::UInt(16));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(UInt32, DataType::UInt(32));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(UInt64, DataType::UInt(64));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Float8, DataType::Float(8));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Float16, DataType::Float(16));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Float32, DataType::Float(32));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Float64, DataType::Float(64));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int32x4, DataType::Int(32, 4));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int32x8, DataType::Int(32, 8));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Int32x16, DataType::Int(32, 16));
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Boolean, DataType::Bool());
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Handle, DataType::Handle());
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Void, DataType::Void());

#undef TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIR_IR_BUILDER_H_
