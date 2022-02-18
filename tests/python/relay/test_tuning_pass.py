from sqlite3 import OptimizedUnicode
import tvm
from tvm import relax, relay, ir
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
from tvm.relay import Function
from tvm.relay import testing
import numpy as np
from tvm.contrib import graph_executor as runtime
import sys


@tvm.instrument.pass_instrument
class PassTracker:
    def run_before_pass(self, module, info):
        print(f"pass name: {info.name}")


def example(x_shape=(1, 32, 16, 16), channels1=32, channels2=32, channels3=32, channels4=32):
    in_c = x_shape[1]
    x = relay.var("x", shape=x_shape)
    w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
    w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
    w3 = relay.var("w3", shape=(channels3, in_c, 3, 3))
    w4 = relay.var("w4", shape=(channels4, in_c, 1, 1))

    args = [x, w1, w2, w3, w4]
    y1 = relay.nn.conv2d(x, w1)
    y2 = relay.nn.conv2d(x, w2)
    # y3 cannot be combined
    y3 = relay.nn.conv2d(x, w3)
    y4 = relay.nn.conv2d(x, w4)
    y5 = relay.nn.max_pool2d(x)

    c_data = np.empty(x_shape).astype("float32")
    c = relay.const(c_data)
    y6 = relay.add(c, c)
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.add(y6, y6)

    z = relay.Tuple((y1, y2, y3, y4, y5, y6))

    return relay.Function(args, z)


class Pass:
    def __init__(self, required=[]):
        self.required = required

    # Does this pass generate valid IRModule?
    def validate():
        pass


@ir.transform.module_pass(opt_level=1)
class MyHeuristicPass(Pass):
    def __init__():
        pass

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        return relay.transform.FoldConstant()(mod)


class TuningPass(Pass):
    def __init__(self, eval_passes):
        super().__init__()
        self.eval_passes = eval_passes

    def evaluate(self, ctx, candidates, num=20, repeat=20):
        target, dev = "llvm", tvm.cpu()
        # Evaluation
        scoreboard = {}
        for candidate in candidates:
            # Apply pass group before build
            seq = tvm.transform.Sequential(self.eval_passes)
            transformed_candidate = seq(candidate)

            # Build candidate
            with tvm.transform.PassContext(opt_level=0):
                lib = relay.build(transformed_candidate, target=target)

            # Measure
            runtime_mod = runtime.GraphModule(lib["default"](dev))
            ftimer = runtime_mod.module.time_evaluator("run", dev, number=num, repeat=repeat)
            perfs = np.array(ftimer().results) * 1000

            # Store transformed candidate
            assert transformed_candidate not in scoreboard
            scoreboard[transformed_candidate] = tuple([np.mean(perfs), np.std(perfs)])

            print(f"   - {candidate}: {np.mean(perfs)}ms\n")

        return scoreboard

    @staticmethod
    def query_cost_model(candidates):
        pass

    @staticmethod
    def select_best_candidate(scoreboard):
        best_perf, best_mod = sys.maxsize, None
        for candidate, (avg, std) in scoreboard.items():
            # Select best one
            if best_perf > avg:
                best_perf = avg
                best_mod = candidate
        return best_perf, best_mod


@ir.transform.module_pass(opt_level=0)
class MyTuningPass1(TuningPass):
    def __init__(self, eval_passes=[]):
        super().__init__(eval_passes)

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        new_mod = relay.transform.InferType()(mod)
        new_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(new_mod)
        candidate_pool = [mod, new_mod]
        scoreboard = self.evaluate(ctx, candidate_pool)
        best_perf, best_mod = self.select_best_candidate(scoreboard)
        return best_mod


@ir.transform.module_pass(opt_level=0)
class MyTuningPass2(TuningPass):
    def __init__(self, eval_passes=[]):
        super().__init__(eval_passes)

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        mod = relay.transform.InferType()(mod)
        new_mod = relay.transform.CombineParallelConv2D(min_num_branches=2)(mod)
        candidate_pool = [mod, new_mod]
        scoreboard = self.evaluate(ctx, candidate_pool)
        best_perf, best_mod = self.select_best_candidate(scoreboard)
        return best_mod


f = example()
mod = tvm.IRModule.from_expr(f)

# Enable joint optimization
# Currently, joint-optimization is happening in tornament manner.
# For example, if we have two tuning passes with two choices each, it will generate 4 candidates in total.
# Then, total number of evals = 4 + 2 (re-evalauate the chosen candidate at outer tuning pass) = 6
# These redundant evaluations can be avoided by passing a status flag later.
# Due to the measurement noise, this re-evaluation may look quite different from original evaluation

custom_pass = MyTuningPass2(
    eval_passes=[MyTuningPass1(eval_passes=[relay.transform.FoldConstant()])]
)

mod = custom_pass(mod)

print("==== optimized ===")
print(mod)
