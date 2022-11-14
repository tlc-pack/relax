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

/*! \brief A temporary Tensor type for `R.Tensor` in ir_builder. */
class TensorTypeNode : public runtime::Object {
 public:
  /*! \brief The type, usually is DynTensorType */
  tvm::relax::DynTensorType type;
  /*! \brief The shape, which is optional. */
  Optional<tvm::relax::Expr> shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("type", &type);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.TensorType";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorTypeNode, runtime::Object);
};

class TensorType : public runtime::ObjectRef {
 public:
  TVM_DLL explicit TensorType(tvm::relax::DynTensorType type, Optional<tvm::relax::Expr> shape);

  TVM_DEFINE_OBJECT_REF_METHODS(TensorType, ObjectRef, TensorTypeNode);
};

/*!
 * \brief Create a TensorType for a DynTensor.
 * \param shape The shape of the tensor. It's runtime dependent if `shape` is None.
 * \param dtype The element data type of the tensor. It's runtime dependent if `dtype` is None.
 * \param ndim The number of dimensions of the tensor. It's runtime dependent if `ndim` is -1.
 * \return The TensorType that is only used in ir_builder.
 */
TVM_DLL TensorType Tensor(Optional<Array<PrimExpr>> shape, DataType dtype, int ndim = -1);

/////////////////////////////// Function ////////////////////////////////

/*!
 * \brief Start a function frame.
 * \return The created ir_builder Function frame.
 */
TVM_DLL FunctionFrame Function();

/*!
 * \brief Add a parameter to the last function frame.
 * \param name The name of the parameter.
 * \param type The type of the parameter.
 * \param shape The shape of the parameter.
 * \return The created function parameter var.
 */
TVM_DLL tvm::relax::Var Arg(const String& name, const Type& type, const tvm::relax::Expr& shape);

/*!
 * \brief Specify the name of the last function frame.
 * \param name The function name.
 */
TVM_DLL void FuncName(const String& name);

/*!
 * \brief Specify the attrs of the last function frame.
 * \param attrs The function attrs.
 */
TVM_DLL void FuncAttrs(Map<String, ObjectRef> attrs);

/*!
 * \brief Specify the return type of the last function frame.
 * \param ret_type The return type. Note: it's a standard `tvm::Type` instead of TensorType.
 */
TVM_DLL void FuncRetType(tvm::Type ret_type);

/*!
 * \brief Specify the return shape of the last function frame.
 * \param ret_shape The return shape.
 */
TVM_DLL void FuncRetShape(tvm::relax::Expr ret_shape);

/*!
 * \brief Specify the return value of the last function frame.
 * \param value The return value.
 */
TVM_DLL void FuncRetValue(const tvm::relax::Expr& value);

///////////////////////////// BindingBlock //////////////////////////////

/*!
 * \brief Start a binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BlockFrame BindingBlock();

/*!
 * \brief Start a dataflow binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BlockFrame Dataflow();

/*!
 * \brief Expose the dataflow block output variables as global ones
 * \param vars The output variables of a dataflow block
 */
TVM_DLL void DataflowBlockOutput(const Array<tvm::relax::Var>& vars);

////////////////////////////// Bindings ////////////////////////////////

/*!
 * \brief Emit a binding to the last binding block frame.
 * \param value The right side value of the bindings to be emitted.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::relax::Var Emit(const tvm::relax::Expr& value);

/*!
 * \brief Emit a match_shape binding to the last binding block frame.
 * \param value The value of the MatchShape to be emitted.
 * \param pattern The pattern of the MatchShape to be emitted.
 * \param emit_var A boolean indicating if the MatchShape contains the emitted variable.
 * \return The emitted var if `emit_var` is true. Otherwise, return `NullOpt`.
 */
TVM_DLL Optional<tvm::relax::Var> EmitMatchShape(const tvm::relax::Expr& value,   //
                                                 const Array<PrimExpr>& pattern,  //
                                                 bool emit_var);

///////////////////////////// Type Deduce //////////////////////////////

/*!
 * \brief Annotate and check the type and shape of relax var.
 * \param var The input var to be annotated.
 * \param anno_type The annotated type.
 * \param anno_shape The annotated shape, which can be undefined.
 * \note This function will check if the type of var is compatible with the annotated type.
 * And we annotate to the var with more detailed type.
 */
TVM_DLL void AnnotateTypeShape(const tvm::relax::Var& var, const Type& anno_type,
                               const Optional<tvm::relax::Expr>& anno_shape);

///////////////////////////// If Then Else /////////////////////////////

/*!
 * \brief Create an if statement.
 * \param condition The condition of if statement.
 * \return The result IfFrame.
 */
IfFrame If(tvm::relax::Expr condition);
/*!
 * \brief Create a then.
 * \return The result ThenFrame.
 */
ThenFrame Then();
/*!
 * \brief Create an else.
 * \return The result ElseFrame.
 */
ElseFrame Else();

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_
