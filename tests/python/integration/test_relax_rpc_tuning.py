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
"""Test tuning a model in Relax over RPC, end-to-end."""
import os
import subprocess
import time

import tvm
from tvm.contrib import utils
import tvm.testing


@tvm.testing.slow
def test_relax_auto_tir_e2e_rpc():
    """
    Run the e2e_auto_tir Relax example script over RPC on localhost.
    """
    rpc_host = "127.0.0.1"
    rpc_key = "Test1"
    rpc_port = "5555"

    tuning_dir = utils.tempdir()

    try:
        tracker_proc = subprocess.Popen(
            [
                "python3",
                "-m",
                "tvm.exec.rpc_tracker",
                "--host",
                rpc_host,
                "--port",
                rpc_port,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # dirty hack, ensure that the tracker has time to start before starting the server
        time.sleep(1)
        server_proc = subprocess.Popen(
            [
                "python3",
                "-m",
                "tvm.exec.rpc_server",
                "--key",
                rpc_key,
                "--tracker",
                f"{rpc_host}:{rpc_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        # timeout is set to 5 because the script tries again every 5s if it fails;
        # we will only permit it one try
        check = subprocess.run(
            [
                "python3",
                "-m",
                "tvm.exec.query_rpc_tracker",
                "--host",
                rpc_host,
                "--port",
                rpc_port,
            ],
            check=True,
            timeout=5,
            capture_output=True,
        )
        # if the key isn't in the printed message, then they didn't connect
        check_output = str(check.stdout)
        assert "Test1" in check_output, check_output

        run_script = subprocess.run(
            [
                "python3",
                os.path.join(os.environ["TVM_HOME"], "apps", "relax_examples", "e2e_auto_tir.py"),
                "--workload",
                "resnet_50",
                "--target",
                # metascheduler requires specifying the number of cores;
                # this uses 16 because that is what is used in the other tuning tests
                "llvm -num-cores 16",
                "--input-shape",
                "[1, 3, 224, 224]",
                # 0 trials so there is no tuning, just testing
                "--num-trials",
                "0",
                "--rpc-host",
                rpc_host,
                "--rpc-port",
                rpc_port,
                "--rpc-key",
                rpc_key,
                "--work-dir",
                tuning_dir.path,
                # this can take several minutes and the default timeout is seldom enough
                "--rpc-timeout",
                "600",
            ],
            check=False,
            capture_output=True,
        )
        # just checking that it completes successfully
        assert run_script.returncode == 0, (run_script.stdout, run_script.stderr)
    finally:
        tracker_proc.terminate()
        server_proc.terminate()


if __name__ == "__main__":
    tvm.testing.main()
