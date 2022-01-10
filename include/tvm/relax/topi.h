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
 * \file relax/topi.h
 */
#ifndef TVM_RELAX_TOPI__H_
#define TVM_RELAX_TOPI__H_

#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/detail/ravel_unravel.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace tvm {
namespace relax {
namespace topi {

// TODO(@yuchen): will need to dedup with functions in Relay when we upstream
/*!
 * \brief GetReduceAxes, get the new axis from indim and other arguments
 * \param indim Number of dimensions of input data.
 * \param axis The input axis vector.
 * \param exclude Whether 'axis' input given is the excluded axis.
 * \return r_axes The new reduced axes of the output.
 */
inline std::vector<int64_t> GetReduceAxes(const uint32_t indim, const Array<Integer>& inaxis,
                                          bool exclude) {
  if (!inaxis.defined()) {
    std::vector<int64_t> r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  std::vector<int64_t> in_axes;
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + indim;
    }

    // Check out of bounds error
    ICHECK(axis >= 0) << "Axis out of bounds in reduce operator.";
    ICHECK(axis < indim) << "Axis out of bounds in reduce operator.";
    in_axes.push_back(axis);
  }

  ICHECK(in_axes[in_axes.size() - 1] < indim)
      << "Reduction axis " << in_axes[in_axes.size() - 1] << " exceeds input dimensions " << indim;

  std::sort(in_axes.begin(), in_axes.end());

  if (!exclude) {
    return in_axes;
  }

  auto r_size = indim - in_axes.size();
  std::vector<int64_t> r_axes(r_size);
  for (uint32_t i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axes.size() && in_axes[j] == i) {
      ++j;
      continue;
    }
    r_axes[k++] = i;
  }
  return r_axes;
}

// Get axis under exclude condition.
Array<Integer> GetExcludeAxes(size_t indim, const Array<Integer>& inaxis) {
  ICHECK(inaxis.defined()) << "Cannot set exclude when axis=None";
  std::vector<bool> axis_flag(indim, true);
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + static_cast<int64_t>(indim);
    }
    // Check out of bounds error
    ICHECK_GE(axis, 0) << "Axis out of bounds in reduce operator.";
    ICHECK_LT(axis, static_cast<int64_t>(indim)) << "Axis out of bounds in reduce operator.";
    axis_flag[axis] = false;
  }

  Array<Integer> r_axes;

  for (size_t i = 0; i < axis_flag.size(); ++i) {
    if (axis_flag[i]) {
      r_axes.push_back(static_cast<int>(i));
    }
  }
  return r_axes;
}

template <typename F>
Array<te::Tensor> ReduceCompute(const te::Tensor& data, const Array<Integer>& axis, bool keepdims,
                                bool exclude, F f) {
  if (data->shape.size() == 0) {
    return {tvm::topi::identity(data)};
  }
  auto axes = axis;
  if (exclude) {
    axes = GetExcludeAxes(data->shape.size(), axis);
    if (axes.size() == 0) {
      return {tvm::topi::identity(data)};
    }
  }

  return {f(data, axes, keepdims, false)};
}

/*!
 * \brief Creates mean operation over given axis.
 * \return A Tensor whose op member is the mean operation
 */
inline te::Tensor mean(const te::Tensor& data, const Array<Integer>& axis, bool keepdims = false,
                       bool exclude = false) {
  PrimExpr count = tir::make_const(data->dtype, 1);
  for (int64_t i : GetReduceAxes(data->shape.size(), axis, exclude)) {
    count *= data->shape[i];
  }
  // Although count is created as inputs[0]->dtype,
  // its type may be changed (promoted) during multiplication
  count = cast(data->dtype, count);
  auto res = ReduceCompute(data, axis, keepdims, exclude, tvm::topi::sum);
  return tvm::topi::divide(res[0], count);
}

inline te::Tensor variance(const te::Tensor& data, const te::Tensor& mean,
                           const Array<Integer>& axis, bool keepdims = false, bool exclude = false,
                           bool unbiased = false) {
  PrimExpr count = tir::make_const(data->dtype, 1);
  auto axes = axis;
  for (int64_t i : GetReduceAxes(data->shape.size(), axis, exclude)) {
    count *= data->shape[i];
  }
  if (unbiased) {
    count -= 1;
  }
  std::vector<Integer> expand_shape;
  auto diff = tvm::topi::subtract(data, mean);
  auto sq_diff = tvm::topi::multiply(diff, diff);
  if (exclude) {
    axes = GetExcludeAxes(sq_diff->shape.size(), axis);
    ICHECK_NE(axes.size(), 0);
  }
  auto var = tvm::topi::divide(tvm::topi::sum(sq_diff, axes, keepdims, false), count);

  return var;
}

Array<PrimExpr> InferNewShape(const Array<PrimExpr>& data_shape, const Array<PrimExpr>& new_shape,
                              bool reverse) {
  Array<PrimExpr> oshape;
  Array<PrimExpr> ishape;
  Array<Integer> newshape;

  if (reverse) {
    ishape.Assign(data_shape.rbegin(), data_shape.rend());
    newshape.Assign(new_shape.rbegin(), new_shape.rend());
  } else {
    ishape = data_shape;
    newshape.Assign(new_shape.begin(), new_shape.end());
  }

  std::unordered_set<size_t> used_input_dims;
  std::unordered_set<size_t> used_output_dims;
  size_t src_idx = 0;
  int infer_idx = -1;

  for (size_t i = 0; i < newshape.size(); ++i) {
    int svalue = newshape[i]->value;
    // special flag handling for shape inference.
    if (svalue > 0) {
      oshape.push_back(newshape[i]);
      ++src_idx;
    } else if (svalue == 0) {
      // keep same
      ICHECK_LT(src_idx, ishape.size());
      used_input_dims.insert(src_idx);
      used_output_dims.insert(oshape.size());
      oshape.push_back(ishape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      ICHECK_LT(infer_idx, 0) << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < ishape.size()) {
        used_input_dims.insert(src_idx);
        used_output_dims.insert(oshape.size());
        oshape.push_back(ishape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      ICHECK_LT(src_idx + 1, ishape.size());
      used_input_dims.insert(src_idx);
      PrimExpr d1 = ishape[src_idx++];
      used_input_dims.insert(src_idx);
      PrimExpr d2 = ishape[src_idx++];
      used_output_dims.insert(oshape.size());
      oshape.push_back(d1 * d2);
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      ICHECK_LT(i + 2, newshape.size());
      ICHECK_LT(src_idx, ishape.size());
      used_input_dims.insert(src_idx);
      PrimExpr d0 = ishape[src_idx++];
      Integer d1 = newshape[++i];
      Integer d2 = newshape[++i];
      if (d1->value == -1) {
        ICHECK_NE(d2->value, -1) << "Split dims cannot both be -1.";
        used_output_dims.insert(oshape.size());

        oshape.push_back(indexdiv(d0, d2));
        used_output_dims.insert(oshape.size());
        oshape.push_back(d2);
      } else {
        used_output_dims.insert(oshape.size());
        oshape.push_back(d1);
        used_output_dims.insert(oshape.size());
        if (d2->value == -1) {
          oshape.push_back(indexdiv(d0, d1));
        } else {
          oshape.push_back(d2);
        }
      }
    } else {
      LOG(FATAL) << "Unsupported special value: " << svalue;
    }
  }

  if (infer_idx >= 0) {
    PrimExpr infer_dim = 1;
    for (size_t i = 0; i < ishape.size(); ++i) {
      if (used_input_dims.count(i) != 0) {
        continue;
      }
      infer_dim *= ishape[i];
    }

    for (size_t i = 0; i < oshape.size(); ++i) {
      if (used_output_dims.count(i) != 0) {
        continue;
      }
      infer_dim = indexdiv(infer_dim, oshape[i]);
    }

    arith::Analyzer ana;
    infer_dim = ana.Simplify(infer_dim);
    oshape.Set(infer_idx, infer_dim);
  }

  if (reverse) {
    Array<PrimExpr> reverse_oshape;
    reverse_oshape.Assign(oshape.rbegin(), oshape.rend());
    return reverse_oshape;
  } else {
    return oshape;
  }
}

te::Tensor reshape(const te::Tensor& data, Array<PrimExpr> newshape, bool reverse = false) {
  auto inferred_newshape = InferNewShape(data->shape, newshape, reverse);
  return tvm::topi::reshape(data, inferred_newshape);
}

/*!
 * \brief Creates an operation that calculates data + bias
 *
 * \param data Tensor with shape [batch, in_dim]
 * \param bias Tensor with shape [batch].
 * \param axis The axis to add the bias to.
 * \return Tensor with shape [batch, in_dim]
 */
inline te::Tensor bias_add(const te::Tensor& data, const te::Tensor& bias, int axis) {
  int data_ndim = data->shape.size();
  if (axis < 0) {
    axis += data_ndim;
  }
  int num_newaxis = data_ndim - axis - 1;
  return tvm::topi::add(data, (num_newaxis ? tvm::topi::expand_dims(bias, 1, num_newaxis) : bias));
}

inline te::Tensor collapse_sum(const te::Tensor& data, Array<PrimExpr> target_shape) {
  ICHECK_GE(data->shape.size(), target_shape.size());
  auto ishape = tvm::topi::detail::GetConstIntValues(data->shape, "ishape");
  auto oshape = tvm::topi::detail::GetConstIntValues(target_shape, "oshape");

  std::vector<int> reduce_axes;
  std::vector<int> squeeze_axes;
  for (int i_ax = ishape.size() - 1, o_ax = oshape.size() - 1; i_ax >= 0; --i_ax) {
    if (o_ax >= 0 && ishape[i_ax] == oshape[o_ax]) {
      --o_ax;
      continue;
    }
    reduce_axes.push_back(i_ax);
    if (o_ax < 0) {  // squeeze o_ax if was added during expansion
      squeeze_axes.push_back(i_ax);
    } else if (oshape[o_ax] == 1) {
      --o_ax;
    }
  }

  if (reduce_axes.size() == 0) return tvm::topi::identity(data, "tensor", "comm_reduce");

  std::reverse(reduce_axes.begin(), reduce_axes.end());
  std::reverse(squeeze_axes.begin(), squeeze_axes.end());
  return tvm::topi::DoCommReduce(data, tvm::sum, target_shape, reduce_axes, squeeze_axes);
}

}  // namespace topi
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TOPI__H_
