/*
 * Templates needed:
 *
 * {{Layout}}
 * {{LeadingDim}}
 * {{DTypeDef}}
 * {{OperatorDef}}
 * {{OperatorName}}
 * {{FUNC_NAME}}
 *
 * Flags:
  nvcc ./apps/vtx/gemm.cu -o /tmp/packaged.so \
  -gencode arch=compute_86,code=sm_86 \
  --shared -O3 -std=c++17 -Xcompiler=-fPIC \
  -Xcompiler=-fno-strict-aliasing \
  -I/root/Projects/tvm-dev/include \
  -I/root/Projects/tvm-dev/3rdparty/dlpack/include \
  -I/root/Projects/tvm-dev/3rdparty/dmlc-core/include \
  -I/root/Projects/tvm-dev/3rdparty/cutlass/include \
  -I/root/Projects/tvm-dev/3rdparty/cutlass/tools/util/include/
 */
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

namespace {

using namespace tvm;
using namespace tvm::runtime;

void _{{FUNC_NAME}}(NDArray A, NDArray B, NDArray C) {
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  // Step 1. Extract M, N, K; layout = 1/0  ===> row/col major
  {{Layout}}
  int m_a = A->shape[layout_a], k_a = A->shape[1 ^ layout_a];
  int k_b = B->shape[layout_b], n_b = B->shape[1 ^ layout_b];
  int m_c = C->shape[layout_c], n_c = C->shape[1 ^ layout_c];
  ICHECK_EQ(m_a, m_c);
  ICHECK_EQ(n_b, n_c);
  ICHECK_EQ(k_a, k_b);
  int M = m_a;
  int N = n_b;
  int K = k_a;

  // Step 2. Extract leading dim
  {{LeadingDim}}
  ICHECK_EQ(lda, A->shape[1]);
  ICHECK_EQ(ldb, B->shape[1]);
  ICHECK_EQ(ldc, C->shape[1]);

  // Step 3. Pointers
  {{DTypeDef}}
  auto* a = reinterpret_cast<DTypeA*>(A->data);
  auto* b = reinterpret_cast<DTypeB*>(B->data);
  auto* c = reinterpret_cast<DTypeC*>(C->data);

  // Step 4. Launch Op
  {{OperatorDef}}
  Operation_{{OperatorName}} gemm_operator;
  const DTypeC alpha = 1.0;
  const DTypeC beta = 0.0;
  cutlass::Status status = gemm_operator({
      {M, N, K},     //
      {a, lda},      //
      {b, ldb},      //
      {c, ldc},      //
      {c, ldc},      //
      {alpha, beta}  //
  });
  CHECK(status == cutlass::Status::kSuccess);
}
}  // namespace

TVM_DLL_EXPORT_TYPED_FUNC({{FUNC_NAME}}, _{{FUNC_NAME}});
