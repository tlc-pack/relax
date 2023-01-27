# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import tempfile

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.testing import transform
from tvm import meta_schedule as ms

import torch
from torch.nn import Module
import copy


def verify_numeric(model_name, input_data=None, rtol=1e-5, atol=1e-5, use_cpu=False):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    input_data = [] if input_data is None else input_data

    if isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or not input_data.shape:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"

    # Setup Torch run and get its output
    with torch.no_grad():
        torch_input = copy.deepcopy(baseline_input)
        torch_model = copy.deepcopy(baseline_model)
        if not use_cpu and torch.cuda.is_available():
            if isinstance(torch_model, torch.nn.Module):
                torch_model = torch_model.cuda()
            torch_input = [inp.cuda() for inp in torch_input]
        torch_outputs = torch_model(*[input.clone() for input in torch_input])

    if isinstance(torch_outputs, tuple):
        torch_outputs = tuple(out.cpu().numpy() for out in torch_outputs)
    else:
        torch_outputs = (torch_outputs.cpu().numpy(),)

    # Setup TVM run
    input_names = [f"y{idx}" for idx, _ in enumerate(baseline_input)]
    input_infos = dict(
        zip(input_names, [(list(tensor.shape), "float32") for tensor in baseline_input])
    )

    if use_cpu:
        target, dev = tvm.target.Target("llvm --num-cores=64"), tvm.cpu()
    else:
        target, dev = tvm.target.Target("nvidia/geforce-rtx-3070"), tvm.cuda()

    mod = relax.frontend.from_pytorch(baseline_model, input_infos)
    assert relax.analysis.well_formed(mod)

    with tempfile.TemporaryDirectory() as work_dir:
        with tvm.transform.PassContext(opt_level=0):
            mod = transform.LowerWithRelayOpStrategyPass(target)(mod)
            db = ms.relax_integration.tune_relax(
                mod=mod,
                params=None,
                target=target,
                work_dir=work_dir,
                max_trials_global=50,
                task_scheduler="round-robin",
            )
            assert relax.analysis.well_formed(mod)
            ex = ms.relax_integration.compile_relax(db, mod, target, params=None)

    vm = relax.VirtualMachine(ex, dev)
    inputs = [tvm.nd.array(inp.clone().numpy(), dev) for inp in baseline_input]
    outputs = vm["main"](*inputs)

    if not isinstance(outputs, list):
        outputs = [outputs]

    # Compare with torch side results
    for i, torch_output in enumerate(torch_outputs):
        output = outputs[i].numpy()
        tvm.testing.assert_allclose(torch_output, output, rtol=rtol, atol=atol)


@tvm.testing.uses_gpu
def test_forward_add():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Mod1(Module):
        def forward(self, x0, x1):
            # use python builtin op
            return x0 + x1

    class Mod2(Module):
        def forward(self, x0, x1):
            # use torch op
            return torch.add(x0, x1)

    class Mod3(Module):
        def forward(self, x):
            return x + 1

    class Mod4(Module):
        def forward(self, x):
            return torch.add(x, 1)

    # Check if we can handle constant metadata
    class Mod5(Module):
        def __init__(self, input_shape):
            super().__init__()
            self.input_shape = input_shape

        def forward(self, x):
            y = torch.ones(self.input_shape, dtype=torch.float)
            if torch.cuda.is_available():
                y = y.cuda()
            return x + y

    input_data = torch.rand(input_shape).float()
    verify_numeric(Mod1(), input_data=[input_data, input_data])
    verify_numeric(Mod2(), input_data=[input_data, input_data])
    verify_numeric(Mod3(), input_data=input_data)
    verify_numeric(Mod4(), input_data=input_data)
    verify_numeric(Mod5(input_shape), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_matmul():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Mod1(Module):
        def forward(self, x0, x1):
            # use python builtin op
            return x0 * x1

    class Mod2(Module):
        def forward(self, x0, x1):
            # use torch op
            return torch.mul(x0, x1)

    class Mod3(Module):
        def forward(self, x):
            return x * 2

    class Mod4(Module):
        def forward(self, x):
            return torch.mul(x, 2)

    # Check if we can handle constant metadata
    class Mod5(Module):
        def __init__(self, input_shape):
            super().__init__()
            self.input_shape = input_shape

        def forward(self, x):
            y = torch.ones(self.input_shape, dtype=torch.float) * 2
            if torch.cuda.is_available():
                y = y.cuda()
            return x * y

    input_data = torch.rand(input_shape).float()
    verify_numeric(Mod1(), input_data=[input_data, input_data])
    verify_numeric(Mod2(), input_data=[input_data, input_data])
    verify_numeric(Mod3(), input_data=input_data)
    verify_numeric(Mod4(), input_data=input_data)
    verify_numeric(Mod5(input_shape), input_data=input_data)


if __name__ == "__main__":
    tvm.testing.main()
