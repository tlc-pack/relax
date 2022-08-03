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
from __future__ import annotations
import os
import subprocess
import time
from typing import Callable, Any

import tvm
from tvm import rpc
from tvm.rpc.tracker import Tracker
from tvm.contrib import utils
import tvm.testing


def retry_with_backoff(thunk: Callable[[], Any]) -> Any:
    """
    Calls the thunk and, if it fails (raises an exception), tries again after a 1s backoff.
    """
    try:
        return thunk()
    except:  # pylint: disable=bare-except
        time.sleep(1.0)
        return thunk()


def check_connection(host: str, port: int, key: str) -> bool:
    """
    Returns true if the tracker at host:port has any servers under the given key
    """
    # Timeout is set to 5 because the script tries again every 5s if it fails;
    # we will only permit it one try.
    # (We could use `rpc.connect_tracker` directly but it retries indefinitely.)
    check = subprocess.check_output(
        [
            "python3",
            "-m",
            "tvm.exec.query_rpc_tracker",
            "--host",
            host,
            "--port",
            str(port),
        ],
        timeout=5,
    )
    # if the key isn't in the printed message, then they didn't connect
    return key in str(check)


def connect_server(host: str, port: int, key: str) -> rpc.Server:
    """
    Starts a server and attempts to connect it to a tracker
    at the given host and port with the given key.

    Subsequently checks if the connection succeeded.
    """
    server = rpc.Server(  # pylint: disable=unused-variable
        host=host, key=key, tracker_addr=(host, port)
    )
    # retry in case we check before the connection comes in
    if not retry_with_backoff(lambda: check_connection(host, port, key)):
        raise Exception("Failed to connect")


@tvm.testing.slow
def test_relax_auto_tir_e2e_rpc():
    """
    Run the e2e_auto_tir Relax example script over RPC on localhost.
    """
    rpc_host = "127.0.0.1"
    rpc_key = "Test1"
    rpc_port = 5555

    # if we don't bind tracker and server to variables, they are deleted and closed
    tracker = Tracker(host=rpc_host, port=rpc_port)  # pylint: disable=unused-variable
    # retry in case the server tries to connect before the tracker starts
    server = retry_with_backoff(  # pylint: disable=unused-variable
        lambda: connect_server(rpc_host, rpc_port, rpc_key)
    )

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
            "--rpc-timeout-sec",
            "600",
        ],
        check=False,
        capture_output=True,
    )
    # just checking that it completes successfully
    assert run_script.returncode == 0, (run_script.stdout, run_script.stderr)


if __name__ == "__main__":
    tvm.testing.main()
