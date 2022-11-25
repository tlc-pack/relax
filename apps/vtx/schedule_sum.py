# pylint: disable=missing-docstring
import tempfile

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T

# pylint: disable=invalid-name


@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(  # type: ignore # pylint: disable=no-self-argument
        A: T.Buffer[(1, 512, 768), "float32"],
        B: T.Buffer[(), "float32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        for i0, i1, i2 in T.grid(1, 512, 768):
            with T.block("reduction"):
                k0, k1, k2 = T.axis.remap("RRR", [i0, i1, i2])
                T.reads(A[k0, k1, k2])
                T.writes(B[()])
                with T.init():
                    B[()] = T.float32(0)  # type: ignore # pylint: disable=no-member
                B[()] = B[()] + A[k0, k1, k2]


def sch_fn(sch: tir.Schedule, bx_decision=None, tx_decision=None) -> None:
    bx = sch.sample_categorical(
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], probs=[0.1] * 11, decision=bx_decision
    )
    (block,) = sch.get_child_blocks(sch.get_block("root"))
    i = sch.fuse(*sch.get_loops(block))
    bx, _ = sch.split(i, factors=[bx, None])
    block = sch.rfactor(bx, 0)
    bx, tx = sch.add_unit_loop(bx), bx
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    tx = sch.sample_categorical(
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], probs=[0.1] * 11, decision=tx_decision
    )
    bx, i = sch.get_loops(block)
    _, tx = sch.split(i, factors=[None, tx])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")


def inject_sum_schedule(extracted_tasks, work_dir):
    tasks = []
    task_weights = []
    for task, logger, rand_state in zip(
        extracted_tasks,
        ms.logging.get_loggers_from_work_dir(work_dir, [t.task_name for t in extracted_tasks]),
        ms.utils.fork_seed(None, n=len(extracted_tasks)),
    ):
        if task.task_name == "sum":
            space = ms.space_generator.ScheduleFn(
                sch_fn=sch_fn,
                sch_rules=[],
            )
        else:
            space = "post-order-apply"
        tasks.append(
            ms.TuneContext(
                mod=task.dispatched[0],
                target=task.target,
                space_generator=space,
                search_strategy="evolutionary",
                task_name=task.task_name,
                logger=logger,
                rand_state=rand_state,
                num_threads="physical",
            ).clone()
        )
        task_weights.append(task.weight)
    return tasks, task_weights


def main():
    target = tvm.target.Target("nvidia/nvidia-t4")
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.tir_integration.tune_tir(
            Module,
            target=target,
            work_dir=work_dir,
            max_trials_global=500,
            space=sch_fn,
        )
    sch = db.query_schedule(Module, target, workload_name="main")
    sch.mod.show()
    print(sch.trace)


def check():
    sch = tir.Schedule(Module)
    sch_fn(sch, bx_decision=7, tx_decision=5)
    mod = tvm.build(sch.mod, target="nvidia/nvidia-t4")

    a_np = np.random.uniform(size=(1, 512, 768), low=-1.0, high=1.0).astype("float32")
    b_np = np.random.uniform(size=()).astype("float32")

    a = tvm.nd.array(a_np, tvm.cuda(0))
    b = tvm.nd.array(b_np, tvm.cuda(0))
    mod(a, b)
    np.testing.assert_allclose(b.numpy(), a_np.sum(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    main()
    # check()
