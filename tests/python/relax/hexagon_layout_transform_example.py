# The original primfunc for convolution prior to hoisting will contain
# fused layout transforms that derive from applying sch.transform_layout
# on cache_read and cache_write blocks. The below is the output we expect
# for two back to back convolutions post hoisting/splitting the transforms
# apart from their respective convolutions into separate primfuncs.


# ----Hoisted and unfused transforms:

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def filter_OIHW_to_tiled_first(
        F: T.Buffer[(256, 64, 3, 3), "int8"], FC_handle: T.handle
    ) -> None:
        FC = T.match_buffer(FC_handle, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        # body
        for ko, co, fh, fw, cio, ki, cii in T.grid(8, 2, 3, 3, 8, 32, 4):
            ci: T.int32 = 4 * cio + cii
            c: T.int32 = 32 * co + ci
            k: T.int32 = 32 * ko + ki
            FC[ko, co, fh, fw, cio, ki, cii] = T.if_then_else(
                0 <= k and k < 256 and 0 <= c and c < 64, F[k, c, fh, fw], 0, dtype="int8"
            )

    @T.prim_func
    def output_tiled_to_NCHW_first(
        BC_handle: T.handle, B: T.Buffer[(1, 256, 45, 39), "int8"]
    ) -> None:
        BC = T.match_buffer(BC_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        # body
        for n, k_1, bh, bw in T.grid(1, 256, 45, 39):
            ko: T.int32 = k_1 // 32
            ki: T.int32 = k_1 % 32
            bho: T.int32 = (bh + 6) // 8
            bhi: T.int32 = (bh + 6) % 8
            bwo: T.int32 = (bw + 0) // 8
            bwi: T.int32 = (bw + 0) % 8
            B[n, k_1, bh, bw] = BC[n, ko, bho, bwo, bhi, bwi, ki]

    @T.prim_func
    def compute_clear_one_accumulator(
        acc: T.Buffer[(2, 8, 8, 32), "int8"], acc_num: T.int32
    ) -> None:
        # body
        for wi, hi, ki_1 in T.grid(8, 8, 32):
            acc[acc_num, hi, wi, ki_1] = T.float32(0)

    @T.prim_func
    def filter_OIHW_to_tiled_second(
        F_1: T.Buffer[(4096, 256, 3, 3), "int8"], FC_handle_1: T.handle
    ) -> None:
        FC_1 = T.match_buffer(
            FC_handle_1, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        for ko_1, co, fh, fw, cio, ki_2, cii in T.grid(128, 8, 3, 3, 8, 32, 4):
            ci_1: T.int32 = 4 * cio + cii
            c_1: T.int32 = 32 * co + ci_1
            k_2: T.int32 = 32 * ko_1 + ki_2
            FC_1[ko_1, co, fh, fw, cio, ki_2, cii] = T.if_then_else(
                0 <= k_2 and k_2 < 4096 and 0 <= c_1 and c_1 < 256,
                F_1[k_2, c_1, fh, fw],
                0,
                dtype="int8",
            )

    @T.prim_func
    def output_tiled_to_NCHW_second(
        BC_handle_1: T.handle, B_1: T.Buffer[(1, 4096, 47, 41), "int8"]
    ) -> None:
        BC_1 = T.match_buffer(
            BC_handle_1, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, k_3, bh, bw in T.grid(1, 4096, 47, 41):
            ko_2: T.int32 = k_3 // 32
            ki_3: T.int32 = k_3 % 32
            bho_1: T.int32 = (bh + 4) // 8
            bhi_1: T.int32 = (bh + 4) % 8
            bwo_1: T.int32 = (bw + 0) // 8
            bwi_1: T.int32 = (bw + 0) % 8
            B_1[n, k_3, bh, bw] = BC_1[n, ko_2, bho_1, bwo_1, bhi_1, bwi_1, ki_3]

    @T.prim_func
    def compute_read_accumulator(
        acc_num_1: T.int32,
        acc_1: T.Buffer[(2, 8, 8, 32), "int8"],
        output: T.Buffer[(1, 32, 8, 8), "int8"],
    ) -> None:
        # body
        for wi, hi, ki_4 in T.grid(8, 8, 32):
            output[0, hi, wi, ki_4] = acc_1[acc_num_1, hi, wi, ki_4]

    @T.prim_func
    def compute_do_conv2d_step(
        main_acc_i: T.int32,
        acc_2: T.Buffer[(2, 8, 8, 32), "int8"],
        AC_croutons: T.Buffer[(2, 8, 8, 32), "int8"],
        F_chunk: T.Buffer[(3, 3, 8, 32, 4), "int8"],
    ) -> None:
        # body
        for bhi_2, awi, fh, fw, cio, ki_5, cii in T.grid(8, 8, 3, 3, 8, 32, 4):
            pseudo_aho: T.int32 = (bhi_2 + fh) // 8
            ahi: T.int32 = (bhi_2 + fh) % 8
            pseudo_bwo: T.int32 = main_acc_i + (awi + (3 - 1) - fw) // 8
            bwi_2: T.int32 = (awi + (3 - 1) - fw) % 8
            acc_num_2: T.int32 = pseudo_bwo % 2
            ci_2: T.int32 = cio * 4 + cii
            A_val: T.int8 = AC_croutons[pseudo_aho, ahi, awi, ci_2]
            F_val: T.int8 = F_chunk[fh, fw, cio, ki_5, cii]
            acc_2[acc_num_2, bhi_2, bwi_2, ki_5] = (
                acc_2[acc_num_2, bhi_2, bwi_2, ki_5] + A_val * F_val
            )

    @T.prim_func
    def compute_second(AC_handle: T.handle, BC_handle_2: T.handle, FC_handle_2: T.handle) -> None:
        AC = T.match_buffer(AC_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        BC_2 = T.match_buffer(
            BC_handle_2, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        FC_2 = T.match_buffer(
            FC_handle_2, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        AC_crouton_ptrs = T.alloc_buffer([2], dtype="handle")
        zero_crouton = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_3, bho_2 in T.grid(128, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_1: T.int32 = awo % 2
                for co in T.serial(8):
                    F_chunk_1: T.handle = T.address_of(
                        FC_2[ko_3, co, 0, 0, 0, 0, 0], dtype="handle"
                    )
                    for nc in T.serial(2):
                        aho: T.int32 = bho_2 - 0 + nc
                        AC_crouton_ptrs[nc] = T.if_then_else(
                            0 <= aho and aho < 7,
                            T.address_of(AC[0, co, aho, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_1,
                            AC_crouton_ptrs.data,
                            F_chunk_1,
                            dtype="int32",
                        )
                    )
                bwo_2: T.int32 = awo - 0
                B_in_bounds: T.bool = bwo_2 >= 0 and bwo_2 < 6
                if B_in_bounds:
                    BC_output_tile: T.handle = T.address_of(
                        BC_2[0, ko_3, bho_2, bwo_2, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator", main_acc_i_1, BC_output_tile, dtype="int32"
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_1, dtype="int32"))

    @T.prim_func
    def compute_clear_all_accumulators(acc_3: T.Buffer[(2, 8, 8, 32), "int8"]) -> None:
        # body
        for acc_num_3, wi, hi, ki_6 in T.grid(2, 8, 8, 32):
            acc_3[acc_num_3, hi, wi, ki_6] = T.float32(0)

    @T.prim_func
    def main(
        A: T.Buffer[(1, 64, 43, 37), "int8"],
        C: T.Buffer[(1, 4096, 47, 41), "int8"],
        F2: T.Buffer[(4096, 256, 3, 3), "int8"],
        F2_handle: T.handle,
    ) -> None:
        F1 = T.buffer_var("int8", "global")
        # body
        # with T.block("root")
        AC_1 = T.alloc_buffer([1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F1C = T.alloc_buffer([8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        BC_first = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        B_2 = T.alloc_buffer([1, 256, 45, 39], dtype="int8")
        BC_second = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F2C = T.alloc_buffer([128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        CC = T.alloc_buffer([1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm")
        T.evaluate(T.call_extern("input_NCHW_to_tiled_first", A.data, AC_1.data, dtype="int8"))
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_first", F1, F1C.data, dtype="int8"))
        T.evaluate(T.call_extern("compute_first", AC_1.data, BC_first.data, F1C.data, dtype="int8"))
        T.evaluate(
            T.call_extern("output_tiled_to_NCHW_first", BC_first.data, B_2.data, dtype="int8")
        )
        T.evaluate(
            T.call_extern("input_NCHW_to_tiled_second", B_2.data, BC_second.data, dtype="int8")
        )
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_second", F2.data, F2C.data, dtype="int8"))
        T.evaluate(T.call_extern("compute_second", BC_second.data, CC.data, F2C.data, dtype="int8"))
        T.evaluate(T.call_extern("output_tiled_to_NCHW_second", CC.data, C.data, dtype="int8"))

    @T.prim_func
    def input_NCHW_to_tiled_first(
        A_1: T.Buffer[(1, 64, 43, 37), "int8"], AC_handle_1: T.handle
    ) -> None:
        AC_2 = T.match_buffer(
            AC_handle_1, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, co, aho_1, awo, ahi_1, awi, ci_3 in T.grid(1, 2, 6, 5, 8, 8, 32):
            c_2: T.int32 = 32 * co + ci_3
            ah: T.int32 = aho_1 * 8 + ahi_1 - 0
            aw: T.int32 = awo * 8 + awi - 0
            AC_2[n, co, aho_1, awo, ahi_1, awi, ci_3] = T.if_then_else(
                0 <= ah and ah < 43 and 0 <= aw and aw < 37,
                A_1[n, c_2, ah, aw],
                T.int8(0),
                dtype="int8",
            )

    @T.prim_func
    def compute_first(AC_handle_2: T.handle, BC_handle_3: T.handle, FC_handle_3: T.handle) -> None:
        AC_3 = T.match_buffer(
            AC_handle_2, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_3 = T.match_buffer(
            BC_handle_3, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        FC_3 = T.match_buffer(
            FC_handle_3, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        AC_crouton_ptrs_1 = T.alloc_buffer([2], dtype="handle")
        zero_crouton_1 = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_4, bho_3 in T.grid(8, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_2: T.int32 = awo % 2
                for co in T.serial(2):
                    F_chunk_2: T.handle = T.address_of(
                        FC_3[ko_4, co, 0, 0, 0, 0, 0], dtype="handle"
                    )
                    for nc in T.serial(2):
                        aho_2: T.int32 = bho_3 - -1 + nc
                        AC_crouton_ptrs_1[nc] = T.if_then_else(
                            0 <= aho_2 and aho_2 < 6,
                            T.address_of(AC_3[0, co, aho_2, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton_1.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_2,
                            AC_crouton_ptrs_1.data,
                            F_chunk_2,
                            dtype="int32",
                        )
                    )
                bwo_3: T.int32 = awo - 0
                B_in_bounds_1: T.bool = bwo_3 >= 0 and bwo_3 < 5
                if B_in_bounds_1:
                    BC_output_tile_1: T.handle = T.address_of(
                        BC_3[0, ko_4, bho_3, bwo_3, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator",
                            main_acc_i_2,
                            BC_output_tile_1,
                            dtype="int32",
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_2, dtype="int32"))

    @T.prim_func
    def input_NCHW_to_tiled_second(
        A_2: T.Buffer[(1, 256, 45, 39), "int8"], AC_handle_3: T.handle
    ) -> None:
        AC_4 = T.match_buffer(
            AC_handle_3, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, co, aho_3, awo, ahi_2, awi, ci_4 in T.grid(1, 8, 7, 5, 8, 8, 32):
            c_3: T.int32 = 32 * co + ci_4
            ah_1: T.int32 = aho_3 * 8 + ahi_2 - 6
            aw_1: T.int32 = awo * 8 + awi - 0
            AC_4[n, co, aho_3, awo, ahi_2, awi, ci_4] = T.if_then_else(
                0 <= ah_1 and ah_1 < 45 and 0 <= aw_1 and aw_1 < 39,
                A_2[n, c_3, ah_1, aw_1],
                T.int8(0),
                dtype="int8",
            )


# ----Hoisted and fused back-to-back padded transforms

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def compute_do_conv2d_step(
        main_acc_i: T.int32,
        acc: T.Buffer[(2, 8, 8, 32), "int8"],
        AC_croutons: T.Buffer[(2, 8, 8, 32), "int8"],
        F_chunk: T.Buffer[(3, 3, 8, 32, 4), "int8"],
    ) -> None:
        # body
        for bhi, awi, fh, fw, cio, ki, cii in T.grid(8, 8, 3, 3, 8, 32, 4):
            pseudo_aho: T.int32 = (bhi + fh) // 8
            ahi: T.int32 = (bhi + fh) % 8
            pseudo_bwo: T.int32 = main_acc_i + (awi + (3 - 1) - fw) // 8
            bwi: T.int32 = (awi + (3 - 1) - fw) % 8
            acc_num: T.int32 = pseudo_bwo % 2
            ci: T.int32 = cio * 4 + cii
            A_val: T.int8 = AC_croutons[pseudo_aho, ahi, awi, ci]
            F_val: T.int8 = F_chunk[fh, fw, cio, ki, cii]
            acc[acc_num, bhi, bwi, ki] = acc[acc_num, bhi, bwi, ki] + A_val * F_val

    @T.prim_func
    def fused_epilogue_prologue(BC_first_handle: T.handle, BC_second_handle: T.handle) -> None:
        BC_first = T.match_buffer(
            BC_first_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_second = T.match_buffer(
            BC_second_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        B = T.alloc_buffer([1, 256, 45, 39], dtype="int8")
        for n, k, bh, bw in T.grid(1, 256, 45, 39):
            ko: T.int32 = k // 32
            ki: T.int32 = k % 32
            bho: T.int32 = (bh + 6) // 8
            bhi: T.int32 = (bh + 6) % 8
            bwo: T.int32 = (bw + 0) // 8
            bwi_1: T.int32 = (bw + 0) % 8
            B[n, k, bh, bw] = BC_first[n, ko, bho, bwo, bhi, bwi_1, ki]
        for n, ko_1, bho_1, bwo_1, bhi_1, bwi_2, ki_1 in T.grid(1, 8, 7, 5, 8, 8, 32):
            k: T.int32 = 32 * ko_1 + ki_1
            bh: T.int32 = bho_1 * 8 + bhi_1 - 6
            bw: T.int32 = bwo_1 * 8 + bwi_2 - 0
            BC_second[n, ko_1, bho_1, bwo_1, bhi_1, bwi_2, ki_1] = T.if_then_else(
                0 <= bh and bh < 45 and 0 <= bw and bw < 39,
                B[n, k, bh, bw],
                T.int8(0),
                dtype="int8",
            )

    @T.prim_func
    def compute_clear_all_accumulators(acc_1: T.Buffer[(2, 8, 8, 32), "int8"]) -> None:
        # body
        for acc_num_1, wi, hi, ki_2 in T.grid(2, 8, 8, 32):
            acc_1[acc_num_1, hi, wi, ki_2] = T.float32(0)

    @T.prim_func
    def compute_first(AC_handle: T.handle, BC_handle: T.handle, FC_handle: T.handle) -> None:
        AC = T.match_buffer(AC_handle, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        BC = T.match_buffer(BC_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        FC = T.match_buffer(FC_handle, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        # body
        # with T.block("root")
        AC_crouton_ptrs = T.alloc_buffer([2], dtype="handle")
        zero_crouton = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_2, bho_2 in T.grid(8, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_1: T.int32 = awo % 2
                for co in T.serial(2):
                    F_chunk_1: T.handle = T.address_of(FC[ko_2, co, 0, 0, 0, 0, 0], dtype="handle")
                    for nc in T.serial(2):
                        aho: T.int32 = bho_2 - -1 + nc
                        AC_crouton_ptrs[nc] = T.if_then_else(
                            0 <= aho and aho < 6,
                            T.address_of(AC[0, co, aho, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_1,
                            AC_crouton_ptrs.data,
                            F_chunk_1,
                            dtype="int32",
                        )
                    )
                bwo_2: T.int32 = awo - 0
                B_in_bounds: T.bool = bwo_2 >= 0 and bwo_2 < 5
                if B_in_bounds:
                    BC_output_tile: T.handle = T.address_of(
                        BC[0, ko_2, bho_2, bwo_2, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator", main_acc_i_1, BC_output_tile, dtype="int32"
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_1, dtype="int32"))

    @T.prim_func
    def output_tiled_to_NCHW_second(
        BC_handle_1: T.handle, B_1: T.Buffer[(1, 4096, 47, 41), "int8"]
    ) -> None:
        BC_1 = T.match_buffer(
            BC_handle_1, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, k_1, bh_1, bw_1 in T.grid(1, 4096, 47, 41):
            ko_3: T.int32 = k_1 // 32
            ki_3: T.int32 = k_1 % 32
            bho_3: T.int32 = (bh_1 + 4) // 8
            bhi_2: T.int32 = (bh_1 + 4) % 8
            bwo_3: T.int32 = (bw_1 + 0) // 8
            bwi_3: T.int32 = (bw_1 + 0) % 8
            B_1[n, k_1, bh_1, bw_1] = BC_1[n, ko_3, bho_3, bwo_3, bhi_2, bwi_3, ki_3]

    @T.prim_func
    def input_NCHW_to_tiled_first(
        A: T.Buffer[(1, 64, 43, 37), "int8"], AC_handle_1: T.handle
    ) -> None:
        AC_1 = T.match_buffer(
            AC_handle_1, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, co, aho_1, awo, ahi_1, awi, ci_1 in T.grid(1, 2, 6, 5, 8, 8, 32):
            c: T.int32 = 32 * co + ci_1
            ah: T.int32 = aho_1 * 8 + ahi_1 - 0
            aw: T.int32 = awo * 8 + awi - 0
            AC_1[n, co, aho_1, awo, ahi_1, awi, ci_1] = T.if_then_else(
                0 <= ah and ah < 43 and 0 <= aw and aw < 37,
                A[n, c, ah, aw],
                T.int8(0),
                dtype="int8",
            )

    @T.prim_func
    def compute_clear_one_accumulator(
        acc_2: T.Buffer[(2, 8, 8, 32), "int8"], acc_num_2: T.int32
    ) -> None:
        # body
        for wi, hi, ki_4 in T.grid(8, 8, 32):
            acc_2[acc_num_2, hi, wi, ki_4] = T.float32(0)

    @T.prim_func
    def filter_OIHW_to_tiled_first(
        F: T.Buffer[(256, 64, 3, 3), "int8"], FC_handle_1: T.handle
    ) -> None:
        FC_1 = T.match_buffer(
            FC_handle_1, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        for ko_4, co, fh, fw, cio, ki_5, cii in T.grid(8, 2, 3, 3, 8, 32, 4):
            ci_2: T.int32 = 4 * cio + cii
            c_1: T.int32 = 32 * co + ci_2
            k_2: T.int32 = 32 * ko_4 + ki_5
            FC_1[ko_4, co, fh, fw, cio, ki_5, cii] = T.if_then_else(
                0 <= k_2 and k_2 < 256 and 0 <= c_1 and c_1 < 64,
                F[k_2, c_1, fh, fw],
                0,
                dtype="int8",
            )

    @T.prim_func
    def compute_read_accumulator(
        acc_num_3: T.int32,
        acc_3: T.Buffer[(2, 8, 8, 32), "int8"],
        output: T.Buffer[(1, 32, 8, 8), "int8"],
    ) -> None:
        # body
        for wi, hi, ki_6 in T.grid(8, 8, 32):
            output[0, hi, wi, ki_6] = acc_3[acc_num_3, hi, wi, ki_6]

    @T.prim_func
    def compute_second(AC_handle_2: T.handle, BC_handle_2: T.handle, FC_handle_2: T.handle) -> None:
        AC_2 = T.match_buffer(
            AC_handle_2, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_2 = T.match_buffer(
            BC_handle_2, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        FC_2 = T.match_buffer(
            FC_handle_2, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        AC_crouton_ptrs_1 = T.alloc_buffer([2], dtype="handle")
        zero_crouton_1 = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_5, bho_4 in T.grid(128, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_2: T.int32 = awo % 2
                for co in T.serial(8):
                    F_chunk_2: T.handle = T.address_of(
                        FC_2[ko_5, co, 0, 0, 0, 0, 0], dtype="handle"
                    )
                    for nc in T.serial(2):
                        aho_2: T.int32 = bho_4 - 0 + nc
                        AC_crouton_ptrs_1[nc] = T.if_then_else(
                            0 <= aho_2 and aho_2 < 7,
                            T.address_of(AC_2[0, co, aho_2, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton_1.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_2,
                            AC_crouton_ptrs_1.data,
                            F_chunk_2,
                            dtype="int32",
                        )
                    )
                bwo_4: T.int32 = awo - 0
                B_in_bounds_1: T.bool = bwo_4 >= 0 and bwo_4 < 6
                if B_in_bounds_1:
                    BC_output_tile_1: T.handle = T.address_of(
                        BC_2[0, ko_5, bho_4, bwo_4, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator",
                            main_acc_i_2,
                            BC_output_tile_1,
                            dtype="int32",
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_2, dtype="int32"))

    @T.prim_func
    def filter_OIHW_to_tiled_second(
        F_1: T.Buffer[(4096, 256, 3, 3), "int8"], FC_handle_3: T.handle
    ) -> None:
        FC_3 = T.match_buffer(
            FC_handle_3, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        for ko_6, co, fh, fw, cio, ki_7, cii in T.grid(128, 8, 3, 3, 8, 32, 4):
            ci_3: T.int32 = 4 * cio + cii
            c_2: T.int32 = 32 * co + ci_3
            k_3: T.int32 = 32 * ko_6 + ki_7
            FC_3[ko_6, co, fh, fw, cio, ki_7, cii] = T.if_then_else(
                0 <= k_3 and k_3 < 4096 and 0 <= c_2 and c_2 < 256,
                F_1[k_3, c_2, fh, fw],
                0,
                dtype="int8",
            )

    @T.prim_func
    def main(
        A_1: T.Buffer[(1, 64, 43, 37), "int8"],
        C: T.Buffer[(1, 4096, 47, 41), "int8"],
        F2: T.Buffer[(4096, 256, 3, 3), "int8"],
        F2_handle: T.handle,
    ) -> None:
        F1 = T.buffer_var("int8", "global")
        # body
        # with T.block("root")
        AC_3 = T.alloc_buffer([1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F1C = T.alloc_buffer([8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        BC_first_1 = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        BC_second_1 = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F2C = T.alloc_buffer([128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        CC = T.alloc_buffer([1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm")
        T.evaluate(T.call_extern("input_NCHW_to_tiled_first", A_1.data, AC_3.data, dtype="int8"))
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_first", F1, F1C.data, dtype="int8"))
        T.evaluate(
            T.call_extern("compute_first", AC_3.data, BC_first_1.data, F1C.data, dtype="int8")
        )
        T.evaluate(
            T.call_extern(
                "fused_epilogue_prologue", BC_first_1.data, BC_second_1.data, dtype="int8"
            )
        )
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_second", F2.data, F2C.data, dtype="int8"))
        T.evaluate(
            T.call_extern("compute_second", BC_second_1.data, CC.data, F2C.data, dtype="int8")
        )
        T.evaluate(T.call_extern("output_tiled_to_NCHW_second", CC.data, C.data, dtype="int8"))


# ----Hoisted with simplification applied to fused back-to-back padded transforms

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer[(1, 64, 43, 37), "int8"],
        C: T.Buffer[(1, 4096, 47, 41), "int8"],
        F2: T.Buffer[(4096, 256, 3, 3), "int8"],
        F2_handle: T.handle,
    ) -> None:
        F1 = T.buffer_var("int8", "global")
        # body
        # with T.block("root")
        AC = T.alloc_buffer([1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F1C = T.alloc_buffer([8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        BC_first = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        BC_second = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F2C = T.alloc_buffer([128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        CC = T.alloc_buffer([1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm")
        T.evaluate(T.call_extern("input_NCHW_to_tiled_first", A.data, AC.data, dtype="int8"))
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_first", F1, F1C.data, dtype="int8"))
        T.evaluate(T.call_extern("compute_first", AC.data, BC_first.data, F1C.data, dtype="int8"))
        T.evaluate(
            T.call_extern("fused_epilogue_prologue", BC_first.data, BC_second.data, dtype="int8")
        )
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_second", F2.data, F2C.data, dtype="int8"))
        T.evaluate(T.call_extern("compute_second", BC_second.data, CC.data, F2C.data, dtype="int8"))
        T.evaluate(T.call_extern("output_tiled_to_NCHW_second", CC.data, C.data, dtype="int8"))

    @T.prim_func
    def compute_clear_one_accumulator(
        acc: T.Buffer[(2, 8, 8, 32), "int8"], acc_num: T.int32
    ) -> None:
        # body
        for wi, hi, ki in T.grid(8, 8, 32):
            acc[acc_num, hi, wi, ki] = T.float32(0)

    @T.prim_func
    def compute_first(AC_handle: T.handle, BC_handle: T.handle, FC_handle: T.handle) -> None:
        AC_1 = T.match_buffer(AC_handle, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        BC = T.match_buffer(BC_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        FC = T.match_buffer(FC_handle, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        # body
        # with T.block("root")
        AC_crouton_ptrs = T.alloc_buffer([2], dtype="handle")
        zero_crouton = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko, bho in T.grid(8, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i: T.int32 = awo % 2
                for co in T.serial(2):
                    F_chunk: T.handle = T.address_of(FC[ko, co, 0, 0, 0, 0, 0], dtype="handle")
                    for nc in T.serial(2):
                        aho: T.int32 = bho - -1 + nc
                        AC_crouton_ptrs[nc] = T.if_then_else(
                            0 <= aho and aho < 6,
                            T.address_of(AC_1[0, co, aho, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i,
                            AC_crouton_ptrs.data,
                            F_chunk,
                            dtype="int32",
                        )
                    )
                bwo: T.int32 = awo - 0
                B_in_bounds: T.bool = bwo >= 0 and bwo < 5
                if B_in_bounds:
                    BC_output_tile: T.handle = T.address_of(
                        BC[0, ko, bho, bwo, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator", main_acc_i, BC_output_tile, dtype="int32"
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i, dtype="int32"))

    @T.prim_func
    def filter_OIHW_to_tiled_first(
        F: T.Buffer[(256, 64, 3, 3), "int8"], FC_handle_1: T.handle
    ) -> None:
        FC_1 = T.match_buffer(
            FC_handle_1, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        for ko, co, fh, fw, cio, ki, cii in T.grid(8, 2, 3, 3, 8, 32, 4):
            ci: T.int32 = 4 * cio + cii
            c: T.int32 = 32 * co + ci
            k: T.int32 = 32 * ko + ki
            FC_1[ko, co, fh, fw, cio, ki, cii] = T.if_then_else(
                0 <= k and k < 256 and 0 <= c and c < 64, F[k, c, fh, fw], 0, dtype="int8"
            )

    @T.prim_func
    def output_tiled_to_NCHW_second(
        BC_handle_1: T.handle, B: T.Buffer[(1, 4096, 47, 41), "int8"]
    ) -> None:
        BC_1 = T.match_buffer(
            BC_handle_1, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, k_1, bh, bw in T.grid(1, 4096, 47, 41):
            ko: T.int32 = k_1 // 32
            ki: T.int32 = k_1 % 32
            bho: T.int32 = (bh + 4) // 8
            bhi: T.int32 = (bh + 4) % 8
            bwo_1: T.int32 = (bw + 0) // 8
            bwi: T.int32 = (bw + 0) % 8
            B[n, k_1, bh, bw] = BC_1[n, ko, bho, bwo_1, bhi, bwi, ki]

    @T.prim_func
    def compute_read_accumulator(
        acc_num_1: T.int32,
        acc_1: T.Buffer[(2, 8, 8, 32), "int8"],
        output: T.Buffer[(1, 32, 8, 8), "int8"],
    ) -> None:
        # body
        for wi, hi, ki_1 in T.grid(8, 8, 32):
            output[0, hi, wi, ki_1] = acc_1[acc_num_1, hi, wi, ki_1]

    @T.prim_func
    def filter_OIHW_to_tiled_second(
        F_1: T.Buffer[(4096, 256, 3, 3), "int8"], FC_handle_2: T.handle
    ) -> None:
        FC_2 = T.match_buffer(
            FC_handle_2, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        for ko_1, co, fh, fw, cio, ki_2, cii in T.grid(128, 8, 3, 3, 8, 32, 4):
            ci_1: T.int32 = 4 * cio + cii
            c_1: T.int32 = 32 * co + ci_1
            k_2: T.int32 = 32 * ko_1 + ki_2
            FC_2[ko_1, co, fh, fw, cio, ki_2, cii] = T.if_then_else(
                0 <= k_2 and k_2 < 4096 and 0 <= c_1 and c_1 < 256,
                F_1[k_2, c_1, fh, fw],
                0,
                dtype="int8",
            )

    @T.prim_func
    def compute_second(AC_handle_1: T.handle, BC_handle_2: T.handle, FC_handle_3: T.handle) -> None:
        AC_2 = T.match_buffer(
            AC_handle_1, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_2 = T.match_buffer(
            BC_handle_2, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        FC_3 = T.match_buffer(
            FC_handle_3, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        AC_crouton_ptrs_1 = T.alloc_buffer([2], dtype="handle")
        zero_crouton_1 = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_2, bho_1 in T.grid(128, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_1: T.int32 = awo % 2
                for co in T.serial(8):
                    F_chunk_1: T.handle = T.address_of(
                        FC_3[ko_2, co, 0, 0, 0, 0, 0], dtype="handle"
                    )
                    for nc in T.serial(2):
                        aho_1: T.int32 = bho_1 - 0 + nc
                        AC_crouton_ptrs_1[nc] = T.if_then_else(
                            0 <= aho_1 and aho_1 < 7,
                            T.address_of(AC_2[0, co, aho_1, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton_1.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_1,
                            AC_crouton_ptrs_1.data,
                            F_chunk_1,
                            dtype="int32",
                        )
                    )
                bwo_2: T.int32 = awo - 0
                B_in_bounds_1: T.bool = bwo_2 >= 0 and bwo_2 < 6
                if B_in_bounds_1:
                    BC_output_tile_1: T.handle = T.address_of(
                        BC_2[0, ko_2, bho_1, bwo_2, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator",
                            main_acc_i_1,
                            BC_output_tile_1,
                            dtype="int32",
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_1, dtype="int32"))

    @T.prim_func
    def input_NCHW_to_tiled_first(
        A_1: T.Buffer[(1, 64, 43, 37), "int8"], AC_handle_2: T.handle
    ) -> None:
        AC_3 = T.match_buffer(
            AC_handle_2, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, co, aho_2, awo, ahi, awi, ci_2 in T.grid(1, 2, 6, 5, 8, 8, 32):
            c_2: T.int32 = 32 * co + ci_2
            ah: T.int32 = aho_2 * 8 + ahi - 0
            aw: T.int32 = awo * 8 + awi - 0
            AC_3[n, co, aho_2, awo, ahi, awi, ci_2] = T.if_then_else(
                0 <= ah and ah < 43 and 0 <= aw and aw < 37,
                A_1[n, c_2, ah, aw],
                T.int8(0),
                dtype="int8",
            )

    @T.prim_func
    def compute_do_conv2d_step(
        main_acc_i_2: T.int32,
        acc_2: T.Buffer[(2, 8, 8, 32), "int8"],
        AC_croutons: T.Buffer[(2, 8, 8, 32), "int8"],
        F_chunk_2: T.Buffer[(3, 3, 8, 32, 4), "int8"],
    ) -> None:
        # body
        for bhi_1, awi, fh, fw, cio, ki_3, cii in T.grid(8, 8, 3, 3, 8, 32, 4):
            pseudo_aho: T.int32 = (bhi_1 + fh) // 8
            ahi: T.int32 = (bhi_1 + fh) % 8
            pseudo_bwo: T.int32 = main_acc_i_2 + (awi + (3 - 1) - fw) // 8
            bwi_1: T.int32 = (awi + (3 - 1) - fw) % 8
            acc_num_2: T.int32 = pseudo_bwo % 2
            ci_3: T.int32 = cio * 4 + cii
            A_val: T.int8 = AC_croutons[pseudo_aho, ahi, awi, ci_3]
            F_val: T.int8 = F_chunk_2[fh, fw, cio, ki_3, cii]
            acc_2[acc_num_2, bhi_1, bwi_1, ki_3] = (
                acc_2[acc_num_2, bhi_1, bwi_1, ki_3] + A_val * F_val
            )

    @T.prim_func
    def compute_clear_all_accumulators(acc_3: T.Buffer[(2, 8, 8, 32), "int8"]) -> None:
        # body
        for acc_num_3, wi, hi, ki_4 in T.grid(2, 8, 8, 32):
            acc_3[acc_num_3, hi, wi, ki_4] = T.float32(0)

    @T.prim_func
    def fused_epilogue_prologue_simplified(
        BC_first_handle: T.handle, BC_second_handle: T.handle
    ) -> None:
        BC_first_1 = T.match_buffer(
            BC_first_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_second_1 = T.match_buffer(
            BC_second_handle, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        # body
        for n, ko_3, bho_2, bwo_3, bhi_2, bwi_2, ki_5 in T.grid(1, 8, 7, 5, 8, 8, 32):
            BC_second_1[n, ko_3, bho_2, bwo_3, bhi_2, bwi_2, ki_5] = BC_first_1[
                n, ko_3, bho_2, bwo_3, bhi_2, bwi_2, ki_5
            ]
        for n, ko_4, bho_3, bwo_4, bhi_3, bwi_3, ki_6 in T.grid(1, 8, 7, 5, 8, 8, 32):
            k_3: T.int32 = 32 * ko_4 + ki_6
            bh: T.int32 = bho_3 * 8 + bhi_3 - 6
            bw: T.int32 = bwo_4 * 8 + bwi_3 - 0
            if (
                bho_3 == 0
                and bhi_3 < 6
                or bho_3 == 7 - 1
                and bhi_3 >= 8 - 5
                or bwo_4 == 0
                and bwi_3 < 0
                or bho_3 == 5 - 1
                and bhi_3 >= 8 - 1
            ):
                BC_second_1[n, ko_4, bho_3, bwo_4, bhi_3, bwi_3, ki_6] = 0


# ----Hoisted and elided back-to-back padded transforms

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def filter_OIHW_to_tiled_second(
        F: T.Buffer[(4096, 256, 3, 3), "int8"], FC_handle: T.handle
    ) -> None:
        FC = T.match_buffer(FC_handle, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        # body
        for ko, co, fh, fw, cio, ki, cii in T.grid(128, 8, 3, 3, 8, 32, 4):
            ci: T.int32 = 4 * cio + cii
            c: T.int32 = 32 * co + ci
            k: T.int32 = 32 * ko + ki
            FC[ko, co, fh, fw, cio, ki, cii] = T.if_then_else(
                0 <= k and k < 4096 and 0 <= c and c < 256, F[k, c, fh, fw], 0, dtype="int8"
            )

    @T.prim_func
    def compute_read_accumulator(
        acc_num: T.int32,
        acc: T.Buffer[(2, 8, 8, 32), "int8"],
        output: T.Buffer[(1, 32, 8, 8), "int8"],
    ) -> None:
        # body
        for wi, hi, ki in T.grid(8, 8, 32):
            output[0, hi, wi, ki] = acc[acc_num, hi, wi, ki]

    @T.prim_func
    def compute_clear_all_accumulators(acc_1: T.Buffer[(2, 8, 8, 32), "int8"]) -> None:
        # body
        for acc_num_1, wi, hi, ki in T.grid(2, 8, 8, 32):
            acc_1[acc_num_1, hi, wi, ki] = T.float32(0)

    @T.prim_func
    def output_tiled_to_NCHW_second(
        BC_handle: T.handle, B: T.Buffer[(1, 4096, 47, 41), "int8"]
    ) -> None:
        BC = T.match_buffer(BC_handle, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm")
        # body
        for n, k_1, bh, bw in T.grid(1, 4096, 47, 41):
            ko: T.int32 = k_1 // 32
            ki: T.int32 = k_1 % 32
            bho: T.int32 = (bh + 4) // 8
            bhi: T.int32 = (bh + 4) % 8
            bwo: T.int32 = (bw + 0) // 8
            bwi: T.int32 = (bw + 0) % 8
            B[n, k_1, bh, bw] = BC[n, ko, bho, bwo, bhi, bwi, ki]

    @T.prim_func
    def compute_do_conv2d_step(
        main_acc_i: T.int32,
        acc_2: T.Buffer[(2, 8, 8, 32), "int8"],
        AC_croutons: T.Buffer[(2, 8, 8, 32), "int8"],
        F_chunk: T.Buffer[(3, 3, 8, 32, 4), "int8"],
    ) -> None:
        # body
        for bhi_1, awi, fh, fw, cio, ki_1, cii in T.grid(8, 8, 3, 3, 8, 32, 4):
            pseudo_aho: T.int32 = (bhi_1 + fh) // 8
            ahi: T.int32 = (bhi_1 + fh) % 8
            pseudo_bwo: T.int32 = main_acc_i + (awi + (3 - 1) - fw) // 8
            bwi_1: T.int32 = (awi + (3 - 1) - fw) % 8
            acc_num_2: T.int32 = pseudo_bwo % 2
            ci_1: T.int32 = cio * 4 + cii
            A_val: T.int8 = AC_croutons[pseudo_aho, ahi, awi, ci_1]
            F_val: T.int8 = F_chunk[fh, fw, cio, ki_1, cii]
            acc_2[acc_num_2, bhi_1, bwi_1, ki_1] = (
                acc_2[acc_num_2, bhi_1, bwi_1, ki_1] + A_val * F_val
            )

    @T.prim_func
    def compute_clear_one_accumulator(
        acc_3: T.Buffer[(2, 8, 8, 32), "int8"], acc_num_3: T.int32
    ) -> None:
        # body
        for wi, hi, ki_2 in T.grid(8, 8, 32):
            acc_3[acc_num_3, hi, wi, ki_2] = T.float32(0)

    @T.prim_func
    def input_NCHW_to_tiled_first(
        A: T.Buffer[(1, 64, 43, 37), "int8"], AC_handle: T.handle
    ) -> None:
        AC = T.match_buffer(AC_handle, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        # body
        for n, co, aho, awo, ahi_1, awi, ci_2 in T.grid(1, 2, 6, 5, 8, 8, 32):
            c_1: T.int32 = 32 * co + ci_2
            ah: T.int32 = aho * 8 + ahi_1 - 0
            aw: T.int32 = awo * 8 + awi - 0
            AC[n, co, aho, awo, ahi_1, awi, ci_2] = T.if_then_else(
                0 <= ah and ah < 43 and 0 <= aw and aw < 37,
                A[n, c_1, ah, aw],
                T.int8(0),
                dtype="int8",
            )

    @T.prim_func
    def compute_first(AC_handle_1: T.handle, BC_handle_1: T.handle, FC_handle_1: T.handle) -> None:
        AC_1 = T.match_buffer(
            AC_handle_1, [1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_1 = T.match_buffer(
            BC_handle_1, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        FC_1 = T.match_buffer(
            FC_handle_1, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        AC_crouton_ptrs = T.alloc_buffer([2], dtype="handle")
        zero_crouton = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_1, bho_1 in T.grid(8, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_1: T.int32 = awo % 2
                for co in T.serial(2):
                    F_chunk_1: T.handle = T.address_of(
                        FC_1[ko_1, co, 0, 0, 0, 0, 0], dtype="handle"
                    )
                    for nc in T.serial(2):
                        aho: T.int32 = bho_1 - -1 + nc
                        AC_crouton_ptrs[nc] = T.if_then_else(
                            0 <= aho and aho < 6,
                            T.address_of(AC_1[0, co, aho, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_1,
                            AC_crouton_ptrs.data,
                            F_chunk_1,
                            dtype="int32",
                        )
                    )
                bwo_1: T.int32 = awo - 0
                B_in_bounds: T.bool = bwo_1 >= 0 and bwo_1 < 5
                if B_in_bounds:
                    BC_output_tile: T.handle = T.address_of(
                        BC_1[0, ko_1, bho_1, bwo_1, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator", main_acc_i_1, BC_output_tile, dtype="int32"
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_1, dtype="int32"))

    @T.prim_func
    def compute_second(AC_handle_2: T.handle, BC_handle_2: T.handle, FC_handle_2: T.handle) -> None:
        AC_2 = T.match_buffer(
            AC_handle_2, [1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        BC_2 = T.match_buffer(
            BC_handle_2, [1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm"
        )
        FC_2 = T.match_buffer(
            FC_handle_2, [128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        # with T.block("root")
        AC_crouton_ptrs_1 = T.alloc_buffer([2], dtype="handle")
        zero_crouton_1 = T.alloc_buffer([8, 8, 32], dtype="int8", scope="global.vtcm")
        for ko_2, bho_2 in T.grid(128, 7):
            T.evaluate(T.call_extern("compute_clear_all_accumulators", dtype="int32"))
            for awo in T.serial(5):
                main_acc_i_2: T.int32 = awo % 2
                for co in T.serial(8):
                    F_chunk_2: T.handle = T.address_of(
                        FC_2[ko_2, co, 0, 0, 0, 0, 0], dtype="handle"
                    )
                    for nc in T.serial(2):
                        aho_1: T.int32 = bho_2 - 0 + nc
                        AC_crouton_ptrs_1[nc] = T.if_then_else(
                            0 <= aho_1 and aho_1 < 7,
                            T.address_of(AC_2[0, co, aho_1, awo, 0, 0, 0], dtype="handle"),
                            zero_crouton_1.data,
                            dtype="handle",
                        )
                    T.evaluate(
                        T.call_extern(
                            "compute_do_conv2d_step",
                            main_acc_i_2,
                            AC_crouton_ptrs_1.data,
                            F_chunk_2,
                            dtype="int32",
                        )
                    )
                bwo_2: T.int32 = awo - 0
                B_in_bounds_1: T.bool = bwo_2 >= 0 and bwo_2 < 6
                if B_in_bounds_1:
                    BC_output_tile_1: T.handle = T.address_of(
                        BC_2[0, ko_2, bho_2, bwo_2, 0, 0, 0], dtype="handle"
                    )
                    T.evaluate(
                        T.call_extern(
                            "compute_read_accumulator",
                            main_acc_i_2,
                            BC_output_tile_1,
                            dtype="int32",
                        )
                    )
                T.evaluate(T.call_extern("compute_clear_accumulator", main_acc_i_2, dtype="int32"))

    @T.prim_func
    def main(
        A_1: T.Buffer[(1, 64, 43, 37), "int8"],
        C: T.Buffer[(1, 4096, 47, 41), "int8"],
        F2: T.Buffer[(4096, 256, 3, 3), "int8"],
        F2_handle: T.handle,
    ) -> None:
        F1 = T.buffer_var("int8", "global")
        # body
        # with T.block("root")
        AC_3 = T.alloc_buffer([1, 2, 6, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F1C = T.alloc_buffer([8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        BC_first = T.alloc_buffer([1, 8, 7, 5, 8, 8, 32], dtype="int8", scope="global.vtcm")
        F2C = T.alloc_buffer([128, 8, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm")
        CC = T.alloc_buffer([1, 128, 7, 6, 8, 8, 32], dtype="int8", scope="global.vtcm")
        T.evaluate(T.call_extern("input_NCHW_to_tiled_first", A_1.data, AC_3.data, dtype="int8"))
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_first", F1, F1C.data, dtype="int8"))
        T.evaluate(T.call_extern("compute_first", AC_3.data, BC_first.data, F1C.data, dtype="int8"))
        T.evaluate(T.call_extern("filter_OIHW_to_tiled_second", F2.data, F2C.data, dtype="int8"))
        T.evaluate(T.call_extern("compute_second", BC_first.data, CC.data, F2C.data, dtype="int8"))
        T.evaluate(T.call_extern("output_tiled_to_NCHW_second", CC.data, C.data, dtype="int8"))

    @T.prim_func
    def filter_OIHW_to_tiled_first(
        F_1: T.Buffer[(256, 64, 3, 3), "int8"], FC_handle_3: T.handle
    ) -> None:
        FC_3 = T.match_buffer(
            FC_handle_3, [8, 2, 3, 3, 8, 32, 4], dtype="int8", scope="global.vtcm"
        )
        # body
        for ko_3, co, fh, fw, cio, ki_3, cii in T.grid(8, 2, 3, 3, 8, 32, 4):
            ci_3: T.int32 = 4 * cio + cii
            c_2: T.int32 = 32 * co + ci_3
            k_2: T.int32 = 32 * ko_3 + ki_3
            FC_3[ko_3, co, fh, fw, cio, ki_3, cii] = T.if_then_else(
                0 <= k_2 and k_2 < 256 and 0 <= c_2 and c_2 < 64,
                F_1[k_2, c_2, fh, fw],
                0,
                dtype="int8",
            )
