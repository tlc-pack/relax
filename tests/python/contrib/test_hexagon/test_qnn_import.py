from __future__ import annotations  # must import to defer parsing of annotations
import sys
import pytest
import numpy as np
import onnx
from tvm.relay import testing

import tvm.testing
import tvm
from tvm import te
from tvm import relay, relax
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.hexagon.session import Session
from tvm.script import relax as R, tir as T
from tvm.relax.testing import relay_translator, nn

def test_qnn_import():
    onnx_model = onnx.load("resnet_int8.onnx")
    relay_mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target_hexagon, disabled_pass=[])
    R.parser.pretty_print(relax_mod)


if __name__ == "__main__":
    test_qnn_import()