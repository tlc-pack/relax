import vortex_fp16
from vortex_fp16 import tvm
import numpy as np

model = vortex_fp16.model()

input0 = tvm.nd.array(np.random.rand(1, 512).astype("int32"), tvm.cuda())
input1 = tvm.nd.array(np.random.rand(1, 512).astype("int32"), tvm.cuda())
input2 = tvm.nd.array(np.random.rand(1, 512).astype("int32"), tvm.cuda())

sample_output = model(input0, input1, input2)

model.measure_runtime(input0, input1, input2)