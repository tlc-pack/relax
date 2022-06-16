import pytest
import tvm
from tvm.relax.ir.instrument import WellFormedInstrument


tvm.transform.PassContext.current().override_instruments([WellFormedInstrument()])
