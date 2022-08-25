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
  Type type;
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
  TVM_DLL explicit TensorType(Type type, Optional<tvm::relax::Expr> shape);

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
 * \param type The type and the shape of the parameter.
 * \return The created function parameter var.
 */
TVM_DLL tvm::relax::Var Arg(const String& name, const TensorType& type);

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
 * \brief Specify the return value of the last function frame.
 * \param value The return value.
 */
TVM_DLL void FuncReturn(const tvm::relax::Expr& value);

///////////////////////////// BindingBlock //////////////////////////////

/*!
 * \brief Start a non-dataflow binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BlockFrame BindingBlock();

/*!
 * \brief Start a dataflow binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BlockFrame Dataflow();

////////////////////////////// Bindings ////////////////////////////////

/*!
 * \brief Emit a binding to the last binding block frame.
 * \param value The right side value of the bindings to be emitted.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::relax::Var Emit(const tvm::relax::Expr& value);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_
