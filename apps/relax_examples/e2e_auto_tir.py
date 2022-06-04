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
import os
import json
import argparse
import logging

import numpy as np  # type: ignore
import tvm

from tvm.meta_schedule.testing.relay_workload import get_network


from typing import Dict

import tvm.meta_schedule as ms

from tvm import tir, relay, relax, runtime
from tvm import transform
from tvm.ir.module import IRModule
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator
from tvm.target.target import Target


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=None,
    )
    args.add_argument(
        "--tune-model",
        type=int,
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=180,
    )
    parsed.rpc_workers = parsed.rpc_config.count_num_servers(allow_missing=False)
    parsed.tune_model = False if parsed.tune_model == 0 else True
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def apply_opt_before_tuning(
    relay_mod: IRModule, params: Dict[str, runtime.NDArray], target: Target
):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)

        relax_mod = relay_translator.from_relay(relay_mod["main"], target=target)
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
    return relax_mod


def apply_opt_after_tuning(relax_mod: IRModule, database: ms.database.Database, target: Target):
    with transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyHistoryBest(database, target)(relax_mod)
        # Should be enabled after LayoutRewrite ready
        # if ARGS.target.kind.name != "cuda":
        #     relax_mod = relax.transform.LayoutRewrite()(relax_mod)
        #     relax_mod = relax.transform.FoldConstant()(relax_mod)
    return relax_mod


def f_remote_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input_data):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        repeat=5,
        min_repeat_ms=500,
    )
    print(evaluator(*input_data))


def main():
    relay_mod, params, (input_name, input_shape, input_dtype) = get_network(
        ARGS.workload,
        ARGS.input_shape,
        cache_dir=ARGS.cache_dir,
    )
    print(f"Workload: {ARGS.workload}")
    print(f"  input_name: {input_name}")
    print(f"  input_shape: {input_shape}")
    print(f"  input_dtype: {input_dtype}")

    path_workload = os.path.join(ARGS.work_dir, "database_workload.json")
    path_tuning_record = os.path.join(ARGS.work_dir, "database_tuning_record.json")
    database = ms.database.JSONDatabase(
        path_workload=path_workload,
        path_tuning_record=path_tuning_record,
    )

    # translate the ResNet model from Relay to Relax
    relax_mod = apply_opt_before_tuning(relay_mod, params, target=ARGS.target)
    assert isinstance(relax_mod, tvm.IRModule)

    if ARGS.tune_model:
        ms.tune_relax(
            mod=relax_mod,
            target=ARGS.target,
            config=ms.TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=64,
                max_trials_per_task=ARGS.num_trials,
                max_trials_global=ARGS.num_trials,
            ),
            runner=ms.runner.RPCRunner(
                rpc_config=ARGS.rpc_config,
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=3,
                    repeat=1,
                    min_repeat_ms=100,
                    enable_cpu_cache_flush=False,
                ),
                alloc_repeat=ARGS.alloc_repeat,
                max_workers=ARGS.rpc_workers,
            ),
            database=database,
            work_dir=ARGS.work_dir,
            num_threads=os.cpu_count(),
        )

    relax_mod = apply_opt_after_tuning(relax_mod, database, ARGS.target)
    if input_dtype.startswith("float"):
        input_data = np.random.uniform(size=input_shape).astype(input_dtype)
    else:
        input_data = np.random.randint(low=0, high=10000, size=input_shape, dtype=input_dtype)

    with transform.PassContext(opt_level=3):
        executable = relax.vm.build(mod=relax_mod, target=ARGS.target)

    run_module_via_rpc(
        rpc_config=ARGS.rpc_config,
        lib=executable.mod,
        dev_type=ARGS.target.kind.name,
        args=[input_data],
        continuation=f_remote_measurement,
    )


if __name__ == "__main__":
    main()
