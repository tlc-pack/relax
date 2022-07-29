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
from tvm import rpc
from tvm.rpc.tracker import Tracker
from tvm.contrib import utils
import tvm.testing


@tvm.testing.slow
def test_relax_auto_tir_e2e_rpc():
    """
    Run the e2e_auto_tir Relax example script over RPC on localhost.
    """
    rpc_host = "127.0.0.1"
    rpc_key = "Test1"
    rpc_port = 5555

    Tracker(host=rpc_host, port=rpc_port)
    # nasty hack: avoid race conditions if the server starts before the tracker
    time.sleep(1)
    rpc.Server(host=rpc_host, key=rpc_key, tracker_addr=(rpc_host, rpc_port))
    # also prevent the query process from firing before the server connects
    time.sleep(1)

    # Timeout is set to 5 because the script tries again every 5s if it fails;
    # we will only permit it one try.
    # (We could use `rpc.connect_tracker` directly but that doesn't have a timeout.)
    check = subprocess.run(
        [
            "python3",
            "-m",
            "tvm.exec.query_rpc_tracker",
            "--host",
            rpc_host,
            "--port",
            str(rpc_port),
        ],
        check=True,
        timeout=5,
        capture_output=True,
    )
    # if the key isn't in the printed message, then they didn't connect
    check_output = str(check.stdout)
    assert "Test1" in check_output, check_output

    tuning_dir = utils.tempdir()
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
            str(rpc_port),
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


if __name__ == "__main__":
    tvm.testing.main()
