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

import tvm
import tvm.testing
from tvm.relay import testing
from tvm import relax, relay
from tvm.relax.testing import relay_translator
from tvm import meta_schedule as ms
from tvm.target import Target
import numpy as np
import pytest
import os, shutil
from tvm.meta_schedule.database import JSONDatabase
import tempfile
from tvm.meta_schedule.task_scheduler import RoundRobin


def get_resnet(batch_size, dtype, layout, image_shape):
    relay_mod, params = testing.resnet.get_workload(
        num_layers=18,
        batch_size=batch_size,
        dtype=dtype,
        layout=layout,
        image_shape=image_shape,
    )

    return relay_mod, params


def create_database(workload_file, tuning_record_file):
    os.makedirs(os.path.dirname(workload_file), exist_ok=True)
    database = JSONDatabase(
        path_workload=workload_file,
        path_tuning_record=tuning_record_file,
    )
    return database


def relay_build_and_run(mod, target, dev, params, data):
    dirpath = "relay_tmp"
    db = create_database(f"{dirpath}/workload.json", f"{dirpath}/record.json")
    with tempfile.TemporaryDirectory() as work_dir:
        ex = ms.tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=ms.EvolutionarySearchConfig(
                num_trials_per_iter=32,
                max_trials_per_task=3,
                max_trials_global=300,
            ),
            # TODO: commented out for now due to the error
            # task_scheduler=RoundRobin,
            work_dir=work_dir,
            database=db,
        )

    rt_mod = tvm.contrib.graph_executor.GraphModule(ex["default"](dev))
    rt_mod.set_input("data", data)
    rt_mod.run()
    out = rt_mod.get_output(0).asnumpy()

    # cleanup
    shutil.rmtree(dirpath)
    return ex, rt_mod, out


def relax_build_and_run(mod, target, dev, params, data):
    dirpath = "relax_tmp"
    db = create_database(f"{dirpath}/workload.json", f"{dirpath}/record.json")
    mod = relax.transform.BindParams("main", params)(mod)

    with tempfile.TemporaryDirectory() as work_dir:
        ex = ms.tune_relax(
            mod=mod,
            target=target,
            config=ms.EvolutionarySearchConfig(
                num_trials_per_iter=32,
                max_trials_per_task=3,
                max_trials_global=300,
            ),
            # TODO: commented out for now due to the error
            # task_scheduler=RoundRobin,
            work_dir=work_dir,
            database=db,
        )
    vm = relax.VirtualMachine(ex, dev)
    res = vm["main"](data)
    out = res.numpy()
    # cleanup
    shutil.rmtree(dirpath)
    return ex, vm, out


def verify_e2e_translation(target_str, layout, batch_size, image_shape):
    target = Target(target_str)
    dev = tvm.device(str(target), dev_id=0)
    relay_mod, params = get_resnet(batch_size, "float32", layout, image_shape)
    input_shape = (1, *image_shape)
    data = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32), dev)

    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)

    relay_ex, relay_rt_mod, relay_out = relay_build_and_run(relay_mod, target, dev, params, data)
    relax_ex, relax_rt_mod, relax_out = relax_build_and_run(relax_mod, target, dev, params, data)

    tvm.testing.assert_allclose(relay_out, relax_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skip(reason="take too much time")
@pytest.mark.parametrize(
    "layout, batch_size, image_shape", [("NCHW", 1, (3, 224, 224)), ("NHWC", 1, (224, 224, 3))]
)
def test_verify_e2e_translation_cpu(layout, batch_size, image_shape):
    verify_e2e_translation("llvm --num-cores=16", layout, batch_size, image_shape)


@pytest.mark.skip(reason="take too much time")
@tvm.testing.requires_gpu
@pytest.mark.parametrize(
    "layout, batch_size, image_shape", [("NCHW", 1, (3, 224, 224)), ("NHWC", 1, (224, 224, 3))]
)
def test_verify_e2e_translation_gpu(layout, batch_size, image_shape):
    verify_e2e_translation("cuda", layout, batch_size, image_shape)


def verify_extracted_tasks(target_str, layout, batch_size, image_shape):
    target = Target(target_str)
    relay_mod, params = get_resnet(batch_size, "float32", layout, image_shape)

    relax_mod = relay_translator.from_relay(
        relay_mod["main"],
        target,
        params,
        pass_config={
            "relay.backend.use_meta_schedule": True,
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
        },
    )

    relay_tasks = ms.extract_task_from_relay(
        relay_mod,
        target=target,
        params=params,
        pass_config={
            "relay.backend.use_meta_schedule": True,
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
        },
    )

    relax_tasks = ms.extract_task_from_relax(relax_mod, target=target, params=params)
    assert len(relay_tasks) == len(relax_tasks)
    # TODO: Can we compare extracted tasks as well?


@pytest.mark.parametrize(
    "layout, batch_size, image_shape",
    [
        ("NCHW", 1, (3, 224, 224)),
        ("NHWC", 1, (224, 224, 3)),
    ],
)
def test_verify_extracted_tasks_cpu(layout, batch_size, image_shape):
    verify_extracted_tasks("llvm --num-cores=16", layout, batch_size, image_shape)


@tvm.testing.requires_gpu
@pytest.mark.parametrize(
    "layout, batch_size, image_shape", [("NCHW", 1, (3, 224, 224)), ("NHWC", 1, (224, 224, 3))]
)
def test_verify_extracted_tasks_gpu(layout, batch_size, image_shape):
    verify_extracted_tasks("cuda", layout, batch_size, image_shape)


if __name__ == "__main__":
    pytest.main([__file__])
