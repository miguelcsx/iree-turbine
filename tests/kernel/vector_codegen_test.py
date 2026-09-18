# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
from iree.turbine.aot import export
from iree.turbine.kernel.compiler.ir import SymbolTable, scf_d, stream_d

M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    # This test is using the compiler "the hard way" until we have all of the
    # API layering in place.
    def testIotaFx(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tk.lang.OutputBuffer[M, tkl.index]):
            i = tk.lang.program_id(0)
            secret_value = ((i * (33 - i) + 4) % 8) // 2
            out[i] = secret_value

        with tk.gen.TestLaunchContext():
            out = torch.zeros(17, dtype=torch.int32)

    def testSoftmaxFx(self):
        @tk.gen.thread(M)
        def softmax_kernel(
            input: tk.lang.InputBuffer[M, K, tkl.f32],
            output: tk.lang.OutputBuffer[M, K, tkl.f32],
        ):
            row_index = tk.lang.program_id(0)
            input_row = input[row_index, :]
            numerator = tkl.exp2(input_row - tkl.max(input_row))
            output_row = numerator / tkl.sum(numerator)
            output[row_index, :] = output_row

        with tk.gen.TestLaunchContext():
            input = torch.randn(128, 64, dtype=torch.float32)
            output = torch.zeros(128, 64, dtype=torch.float32)
            softmax_kernel(input, output)

    def testForLoopFx(self):
        @tk.gen.thread(M)
        def for_loop_kernel(
            input: tk.lang.InputBuffer[M, K, tkl.f32],
            output: tk.lang.OutputBuffer[M, K, tkl.f32],
        ):
            row_idx = tkl.program_id(0)
            sum = input[row_idx, 0]
            prefetch = input[row_idx, 1]

            @tkl.for_loop(2, 5, init_args=[sum, prefetch])
            def prefetch_sum(i, sum, prefetch):
                new_sum = sum + prefetch
                new_prefetch = input[row_idx, i]
                return new_sum, new_prefetch

            output[row_idx, 0] = prefetch_sum[0]

        with tk.gen.TestLaunchContext():
            input = torch.randn(128, 64, dtype=torch.float32)
            output = torch.zeros(128, 64, dtype=torch.float32)
            for_loop_kernel(input, output)

    def testGemmFx(self):
        N = tkl.sym.N
        M = tkl.sym.M
        K = tkl.sym.K
        BLOCK_SIZE = tkl.sym.BLOCK_SIZE

        @tk.gen.thread(N // BLOCK_SIZE, M // BLOCK_SIZE)
        def gemm_kernel(
            A: tkl.InputBuffer[N, K, tkl.f32],
            B: tkl.InputBuffer[K, M, tkl.f32],
            output: tkl.OutputBuffer[N, M, tkl.f32],
        ):
            grid_n = tkl.program_id(0)
            grid_m = tkl.program_id(1)

            acc = tkl.constant((BLOCK_SIZE, BLOCK_SIZE), tkl.f32, 0.0)

            @tkl.for_loop(0, K // BLOCK_SIZE, init_args=[acc])
            def body(i, c):
                a = tkl.load(A, (grid_n, i * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
                b = tkl.load(B, (i * BLOCK_SIZE, grid_m), (BLOCK_SIZE, BLOCK_SIZE))
                return (tkl.dot(a, b, c),)

            tkl.store(output, (grid_n, grid_m), body[0])

        with tk.gen.TestLaunchContext({BLOCK_SIZE: 32}):
            A = torch.randn(512, 1024, dtype=torch.float32)
            B = torch.randn(1024, 2048, dtype=torch.float32)
            output = torch.zeros(512, 2048, dtype=torch.float32)
            gemm_kernel(A, B, output)


def _ops_named(block, name):
    """The ops of a block that are called `name`."""
    return [op for op in block.operations if op.operation.name == name]


def _only_child(op, name):
    """The single op called `name` in the only region of `op`."""
    [child] = _ops_named(op.regions[0].blocks[0], name)
    return child


def _nested_ops(op):
    """Every op under `op`, descending the regions by hand."""
    ops = []
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                ops += [child, *_nested_ops(child)]
    return ops


def _kernel_function(module):
    """The kernel function of the exported module, found by its symbol."""
    executable = _only_child(module, "stream.executable")
    [export_op] = _ops_named(
        executable.regions[0].blocks[0], "stream.executable.export"
    )
    inner = _only_child(executable, "builtin.module")
    return SymbolTable(inner.operation)[export_op.attributes["sym_name"].value]


def _entry_block(module):
    """The entry block of the kernel function."""
    return _kernel_function(module).regions[0].blocks[0]


def _is_axis(op, axis):
    """True for the workgroup id of one grid axis."""
    return (
        isinstance(op, stream_d.DispatchWorkgroupIDOp)
        and int(op.attributes["dimension"]) == axis
    )


def _is_subspan(op):
    """True for a buffer subspan."""
    return isinstance(op, stream_d.BindingSubspanOp)


class RegionBindingTest(unittest.TestCase):
    """Checks that a binding the emitter remembers dominates every use of it.

    The id of a grid axis and the subspan of a buffer are created the first time
    the kernel asks for them, and remembered for the rest of the trace.  That
    first ask can happen inside the body of a loop, and the value is asked for
    again at the root afterwards, where a value defined in the loop body does
    not dominate it.
    """

    def _compile(self, kernel):
        """Exports a kernel applied to one buffer and verifies its IR."""

        class Case(torch.nn.Module):
            def forward(self, input):
                return kernel(input)

        module = export(Case(), torch.randn(4, 8)).mlir_module
        module.verify()
        return module

    def _ops_and_loop(self, module):
        """The ops of the kernel entry block and the index of its loop."""
        ops = list(_entry_block(module).operations)
        loop = next(i for i, op in enumerate(ops) if isinstance(op, scf_d.ForOp))
        return ops, loop

    def _assert_hoisted(self, module, matches, what):
        """Every op matching `matches` is in the entry block, before the loop."""
        ops, loop = self._ops_and_loop(module)
        inside = [op for op in _nested_ops(ops[loop]) if matches(op)]
        self.assertEqual(inside, [], f"{what} is materialized inside the loop")
        positions = [i for i, op in enumerate(ops) if matches(op)]
        self.assertTrue(positions, f"the kernel has no {what}")
        self.assertLess(max(positions), loop, f"{what} is not hoisted")

    def _assert_built_from_the_entry_block(self, module):
        """An ABI op is materialized from the entry block and from nothing else.

        Its operands are the arguments of that block, or values that dominate it
        in the same block: that is what makes the top of the block its home.
        """
        block = _entry_block(module)
        defined = set(block.arguments)
        for op in block.operations:
            if isinstance(
                op, (stream_d.BindingSubspanOp, stream_d.DispatchWorkgroupIDOp)
            ):
                self.assertTrue(
                    all(operand in defined for operand in op.operands),
                    f"{op.operation.name} is not built from the entry block",
                )
            defined.update(op.results)

    def testGridAxisIdAskedForInLoopBodyIsUsedAtRoot(self):
        @tk.gen.kernel(M, K)
        def copy_column(
            input: tkl.InputBuffer[M, K, tkl.f32],
            output: tkl.OutputBuffer[M, K, tkl.f32],
        ):
            row_index = tkl.program_id(0)
            value = tkl.load(input, (row_index, 0), (1, 1))

            @tkl.for_loop(0, 4, init_args=[value])
            def body(i, value):
                # Axis 1 is asked for here for the first time, in the region.
                return (tkl.load(input, (row_index, tkl.program_id(1)), (1, 1)),)

            # ...and again here, at the root, after the loop has been emitted.
            tkl.store(output, (row_index, tkl.program_id(1)), body[0])

        module = self._compile(copy_column)

        # The loop body and the root share the one id of that axis.
        ids = [op for op in _nested_ops(_kernel_function(module)) if _is_axis(op, 1)]
        self.assertEqual(len(ids), 1, "the kernel must own one id for axis 1")

        self._assert_hoisted(module, lambda op: _is_axis(op, 1), "id of axis 1")
        self._assert_built_from_the_entry_block(module)

    def testGridAxisIdAskedForInANestedRegionIsUsedAtRoot(self):
        @tk.gen.kernel(M, K)
        def nested_loops(
            input: tkl.InputBuffer[M, K, tkl.f32],
            output: tkl.OutputBuffer[M, K, tkl.f32],
        ):
            row_index = tkl.program_id(0)
            value = tkl.load(input, (row_index, 0), (1, 1))

            @tkl.for_loop(0, 2, init_args=[value])
            def outer(i, value):
                # Axis 1 is asked for here for the first time, two regions deep:
                # the ambient insertion point is the body of the inner loop.
                @tkl.for_loop(0, 2, init_args=[value])
                def inner(j, value):
                    return (tkl.load(input, (row_index, tkl.program_id(1)), (1, 1)),)

                return (inner[0],)

            # ...and again here, at the root, after both loops have been emitted.
            tkl.store(output, (row_index, tkl.program_id(1)), outer[0])

        module = self._compile(nested_loops)
        self._assert_hoisted(module, lambda op: _is_axis(op, 1), "id of axis 1")
        self._assert_built_from_the_entry_block(module)

    def testBufferSubspanAskedForInLoopBodyIsUsedAtRoot(self):
        @tk.gen.kernel(M)
        def stage_row(
            input: tkl.InputBuffer[M, K, tkl.f32],
            output: tkl.OutputBuffer[M, K, tkl.f32],
            scratch: tkl.OutputBuffer[M, K, tkl.f32],
        ):
            row_index = tkl.program_id(0)
            value = tkl.load(input, (row_index, 0), (1, 1))

            @tkl.for_loop(0, 4, init_args=[value])
            def body(i, value):
                # `scratch` is asked for here for the first time, in the region.
                tkl.store(scratch, (row_index, i), value)
                return (value,)

            # ...and again here, at the root, after the loop has been emitted.
            tkl.store(output, (row_index, 0), tkl.load(scratch, (row_index, 0), (1, 1)))

        module = self._compile(stage_row)

        # Every buffer the kernel binds, `scratch` included, is bound in the block.
        buffers = {
            arg
            for arg in _entry_block(module).arguments
            if str(arg.type) == "!stream.binding"
        }
        bound = [
            op.binding
            for op in _nested_ops(_kernel_function(module))
            if _is_subspan(op)
        ]
        self.assertEqual(set(bound), buffers, "not every buffer is bound in the block")

        self._assert_hoisted(module, _is_subspan, "buffer subspan")
        self._assert_built_from_the_entry_block(module)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
