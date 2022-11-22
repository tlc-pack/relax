
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/device_memory.h>

#include <iostream>

#include "./kernel_forward.h"

// clang-format: off

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

// clang-format: on

namespace {

const int num_heads = 12;

int _FusedQKVToCxt(DLTensor* QKV, DLTensor* Mask, DLTensor* Output) {
  using T = float;
  CHECK_EQ(QKV->ndim, 3);     // B, S, (NH + NH + NH')
  CHECK_EQ(Mask->ndim, 2);    // B, S
  CHECK_EQ(Output->ndim, 4);  // B, N, S, H'

  using T = float;
  using Attention = AttentionKernel<T, cutlass::arch::Sm75, /*is_aligned=*/true,
                                    /*queries_per_block=*/64, /*keys_per_block=*/64,
                                    /*single_value_iteration=*/true>;

  typename Attention::Params p;

  int64_t head_size = QKV->shape[2] / 3 / num_heads;
  p.query_ptr = reinterpret_cast<T*>(QKV->data);
  p.key_ptr = reinterpret_cast<T*>(QKV->data) + head_size * num_heads;
  p.value_ptr = reinterpret_cast<T*>(QKV->data) + head_size * num_heads * 2;
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T*>(Output->data);
  static_assert(!Attention::kNeedsOutputAccumulatorBuffer);
  p.output_accum_ptr = nullptr;
  p.mask_ptr = Mask != nullptr ? static_cast<int*>(Mask->data) : nullptr;

  p.num_heads = num_heads;
  p.num_batches = QKV->shape[0];
  p.head_dim = head_size;  // head size
  p.head_dim_value = p.head_dim;
  p.num_queries = QKV->shape[1];  // Q seq len
  p.num_keys = p.num_queries;     // KV seq len
  p.causal = false;

  // strides
  // strideM: stride of the sequence dimension
  // strideH: stride of the num_heads dimension
  // strideB: stride of the batch dimension

  int64_t strideM = (p.head_dim * 2 + p.head_dim_value) * num_heads;
  p.q_strideM = strideM;
  p.k_strideM = strideM;
  p.v_strideM = strideM;

  p.q_strideH = p.head_dim;
  p.k_strideH = p.head_dim;
  p.v_strideH = p.head_dim_value;
  p.o_strideH = p.head_dim_value * p.num_queries;

  p.q_strideB = p.q_strideH * p.num_queries;
  p.k_strideB = p.k_strideH * p.num_keys;
  p.v_strideB = p.v_strideH * p.num_keys;
  p.o_strideB = p.o_strideH * p.num_queries;

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    static bool once = [&]() {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }
  CHECK(Attention::check_supported(p));
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes /*cuda stream?*/>>>(p);
  return 0;
}

}  // namespace

TVM_DLL_EXPORT_TYPED_FUNC(FusedQKVToCxt, _FusedQKVToCxt);
