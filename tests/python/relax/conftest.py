import pytest
import tvm
from tvm import relax


@tvm.instrument.pass_instrument
class WellFormedInstrument:
    """An instrument that checks input/output mod is well formed.
    It will skip specific passes, like Normalize.
    """

    def __init__(self):
        self.skip_pass_name = ["Normalize", "ResolveGlobals"]

    def run_before_pass(self, mod, pass_info):
        if pass_info.name in self.skip_pass_name:
            return
        assert relax.analysis.well_formed(mod)

    def run_after_pass(self, mod, pass_info):
        if pass_info.name in self.skip_pass_name:
            return
        assert relax.analysis.well_formed(mod)


tvm.transform.PassContext.current().override_instruments([WellFormedInstrument()])
