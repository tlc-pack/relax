/*
 * Templates needed:
 *
 * {{Layout}}
 * {{LeadingDim}}
 * {{DTypeDef}}
 * {{OperatorDef}}
 * {{OperatorName}}
 * {{FUNC_NAME}}
 * {{BiasNDArray}}
 * {{BiasExists}}
 * {{BiasPtr}}
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
#include <cutlass/util/device_memory.h>

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

#define CUTLASS_CHECK(status)                                                                  \
{                                                                                              \
  cutlass::Status error = status;                                                              \
  if (error != cutlass::Status::kSuccess) {                                                    \
    std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
              << std::endl;                                                                    \
    exit(EXIT_FAILURE);                                                                        \
  }                                                                                            \
}

void _{{FUNC_NAME}}(NDArray A, NDArray B, {{BiasNDArray}} NDArray C) {
  int a_pad = 1, b_pad = 1, c_pad = 1;
  CHECK_EQ(A->ndim, 2 + a_pad);
  CHECK_EQ(B->ndim, 2 + b_pad);
  CHECK_EQ(C->ndim, 2 + c_pad);

  // Step 1. Extract M, N, K; layout = 0/1  ===> row/col major
  {{Layout}}
  int m_a = A->shape[layout_a + a_pad], k_a = A->shape[(layout_a ^ 1) + a_pad];
  int k_b = B->shape[layout_b + b_pad], n_b = B->shape[(layout_b ^ 1) + b_pad];
  int m_c = C->shape[layout_c + c_pad], n_c = C->shape[(layout_c ^ 1) + c_pad];
  // LOG(INFO) << "layout_a = " << layout_a << ", layout_b = " << layout_b << ", layout_c = " << layout_c;
  // LOG(INFO) << "A->shape = " << A->shape[0] << ", " << A->shape[1] << ", " << A->shape[2];
  // LOG(INFO) << "B->shape = " << B->shape[0] << ", " << B->shape[1] << ", " << B->shape[2];
  // LOG(INFO) << "C->shape = " << C->shape[0] << ", " << C->shape[1] << ", " << C->shape[2];
  // LOG(INFO) << "m_a = " << m_a << ", k_a = " << k_a;
  // LOG(INFO) << "k_b = " << k_b << ", n_b = " << n_b;
  // LOG(INFO) << "m_c = " << m_c << ", n_c = " << n_c;
  ICHECK_EQ(m_a, m_c);
  ICHECK_EQ(n_b, n_c);
  ICHECK_EQ(k_a, k_b);
  int M = m_a;
  int N = n_b;
  int K = k_a;

  // Step 2. Extract leading dim
  {{LeadingDim}}
  ICHECK_EQ(lda, A->shape[1 + a_pad]);
  ICHECK_EQ(ldb, B->shape[1 + b_pad]);
  ICHECK_EQ(ldc, C->shape[1 + c_pad]);

  // Step 3. Pointers
  {{DTypeDef}}
  auto* a = reinterpret_cast<DTypeA*>(A->data);
  auto* b = reinterpret_cast<DTypeB*>(B->data);
  {{BiasPtr}}
  auto* c = reinterpret_cast<DTypeC*>(C->data);

  // Step 4. Launch Op
  {{OperatorDef}}
  using GemmOp = Operation_{{OperatorName}};
  GemmOp gemm_operator;

  const DTypeC alpha{1.0};
  const DTypeC beta{ {{BiasExists}} };
  typename GemmOp::Arguments arguments{
      {M, N, K},     //
      {a, lda},      //
      {b, ldb},      //
      {bias_ptr, 0}, //
      {c, ldc},      //
      {alpha, beta}  //
  };
  cutlass::device_memory::allocation<uint8_t> workspace(GemmOp::get_workspace_size(arguments));
  CUTLASS_CHECK(gemm_operator.can_implement(arguments));
  CUTLASS_CHECK(gemm_operator.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_operator());
}
}  // namespace

TVM_DLL_EXPORT_TYPED_FUNC({{FUNC_NAME}}, _{{FUNC_NAME}});
