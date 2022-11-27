import onnx
from onnx import numpy_helper, helper, TensorProto
import numpy as np

# Iterate through all initializers and replace them with fp16.
model = onnx.load("onnx_emails_int32_dummy_turing_vortex_fixed_v2.onnx")
graph = model.graph

for i, x in enumerate(graph.initializer):
    # Check if this is a float
    if x.data_type == TensorProto.FLOAT:
        # If so cast it to fp16 and replace it.
        name = x.name
        data = numpy_helper.to_array(x)
        # Cast to fp16.
        data = data.astype("float16")
        # Replace in initializer array.
        graph.initializer[i].CopyFrom(numpy_helper.from_array(data, name))

# Also replace output types with float16.
for o in graph.output:
    if o.type.tensor_type.elem_type == TensorProto.FLOAT:
        o.type.tensor_type.elem_type = TensorProto.FLOAT16

# Save the new fp16 graph.
onnx.save(model, "vortex_fp16.onnx")