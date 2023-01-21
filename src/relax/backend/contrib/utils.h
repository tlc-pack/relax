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
 * \file relax/backend/contrib/utils.h
 * \brief Utils function for backend
 */
#ifndef TVM_RELAX_BACKEND_CONTRIB_UTILS_H_
#define TVM_RELAX_BACKEND_CONTRIB_UTILS_H_

#include <dmlc/json.h>
#include <tvm/driver/driver_api.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/target/codegen.h>
#include <tvm/target/virtual_device.h>
#include <tvm/te/operation.h>
#include <tvm/tir/usmp/utils.h>

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../runtime/meta_data.h"
#include "../../../target/metadata.h"
#include "tvm/runtime/ndarray.h"

namespace tvm {
namespace relax {
namespace backend {

/*!
 * \brief A simple wrapper around ExprFunctor for a single argument case.
 *  The result of visit is memoized.
 */
template <typename OutputType>
class MemoizedExprTranslator : public ::tvm::relax::ExprFunctor<OutputType(const Expr&)> {
  using BaseFunctor = ::tvm::relax::ExprFunctor<OutputType(const Expr&)>;

 public:
  /*! \brief virtual destructor */
  virtual ~MemoizedExprTranslator() {}

  /*!
   * \brief The memoized call.
   * \param n The expression node.
   * \return The result of the call
   */
  virtual OutputType VisitExpr(const Expr& n) {
    ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_[n] = res;
    return res;
  }

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, OutputType, ObjectPtrHash, ObjectPtrEqual> memo_;
};

/*!
 * \brief Get the Packed Func
 *
 * \param func_name
 * \return const PackedFunc*
 */
inline const PackedFunc* GetPackedFunc(const std::string& func_name) {
  return tvm::runtime::Registry::Get(func_name);
}

/*!
 * \brief Extract shape from an IndexExpr array to std::vector<int64_t>
 *
 * \param shape The shape in Array
 * \return The converted shape in std::vector<int64_t>
 */

inline std::vector<int64_t> GetIntShape(const Array<PrimExpr>& shape) {
  std::vector<int64_t> ret;
  for (const auto& dim : shape) {
    const int64_t* pval = tir::as_const_int(dim);
    ret.push_back(pval ? *pval : -1);
  }
  return ret;
}

/*!
 * \brief Convert type to string
 *
 * \param typ
 * \return std::string string format of type
 */
inline std::string DType2String(const tvm::DataType dtype) {
  std::ostringstream os;
  if (dtype.is_float()) {
    os << "float";
  } else if (dtype.is_int()) {
    os << "int";
  } else if (dtype.is_uint()) {
    os << "uint";
  } else if (dtype.is_bfloat16()) {
    os << "bfloat";
  } else if ((*GetPackedFunc("runtime._datatype_get_type_registered"))(dtype.code())) {
    os << "custom["
       << (*GetPackedFunc("runtime._datatype_get_type_name"))(dtype.code()).operator std::string()
       << "]";
  } else {
    LOG(FATAL) << "Unknown type with code " << static_cast<unsigned>(dtype.code());
  }
  os << dtype.bits();
  return os.str();
}

}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_CONTRIB_UTILS_H_
