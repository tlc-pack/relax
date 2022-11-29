import onnxruntime
import time
import numpy as np

onnx_providers = onnxruntime.get_available_providers()

print("Available Providers: ", onnx_providers)

input_dict = {
    "q_title_token_ids": np.random.randint(256, size=[1, 512]).astype("int32"),
    "q_title_token_types": np.random.randint(256, size=[1, 512]).astype("int32"),
    "q_title_token_masks": np.random.randint(256, size=[1, 512]).astype("int32"),
}

#session = onnxruntime.InferenceSession("onnx_emails_int32_dummy_turing_vortex_fixed_v2.onnx", providers=["CUDAExecutionProvider"])
session = onnxruntime.InferenceSession("vortex_fp16.onnx", providers=["CUDAExecutionProvider"])

# Warm up run
outputs = session.run([], input_dict)

num_iters = 1000
start = time.time()
for i in range(num_iters):
    outputs = session.run([], input_dict)
end = time.time()

print("Runtime: %f ms" % ((end - start) * 1000 / num_iters))