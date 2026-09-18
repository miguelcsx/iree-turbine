# Copyright 2024 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import unittest

import torch
from iree.turbine.aot import export
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
from iree.turbine.runtime import Launchable


def export_softmax_kernel():
    M = tkl.sym.M
    N = tkl.sym.K

    @tk.gen.kernel(M)
    def softmax(
        input: tkl.InputBuffer[M, N, tkl.f16], output: tkl.OutputBuffer[M, N, tkl.f16]
    ):
        row_index = tkl.program_id(0)
        row = tkl.load(input, (row_index, 0), (1, N))
        row_minus_max = row - tkl.max(row)
        numerator = tkl.exp2(row_minus_max)
        denominator = tkl.sum(numerator)
        softmax_output = numerator / denominator
        tkl.store(output, (row_index, 0), softmax_output)

    class NN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64, dtype=torch.float16)

        def forward(self, x):
            x = self.linear(x)
            x = softmax(x)
            return x

    model = NN()
    a = torch.ones(64, 64, dtype=torch.float16)
    exported = export(model, a)
    return exported


def export_copy_column_kernel(input):
    """Exports a kernel whose grid id is first asked for inside a loop body.

    Axis 1 selects the element an instance owns.  It is first used in the body
    of the loop and used again at the root, where only a value materialized at
    the top of the entry block dominates it.
    """
    M = tkl.sym.M
    K = tkl.sym.K

    @tk.gen.kernel(M, K)
    def copy_column(
        input: tkl.InputBuffer[M, K, tkl.f32], output: tkl.OutputBuffer[M, K, tkl.f32]
    ):
        row_index = tkl.program_id(0)
        value = tkl.load(input, (row_index, 0), (1, 1))

        @tkl.for_loop(0, 4, init_args=[value])
        def body(i, value):
            return (tkl.load(input, (row_index, tkl.program_id(1)), (1, 1)),)

        tkl.store(output, (row_index, tkl.program_id(1)), body[0])

    class NN(torch.nn.Module):
        def forward(self, input):
            return copy_column(input)

    return export(NN(), input)


class LaunchTest(unittest.TestCase):
    def test_kernel_binding_a_grid_axis_in_a_loop_body(self):
        input = torch.randn(4, 8, dtype=torch.float32)
        asm = str(export_copy_column_kernel(input).mlir_module)
        output = Launchable.jit_compile(asm, entry_point="main")(input)

        # Every element is written by the instance that owns it, so the result
        # does not depend on the order the workgroups run in.
        torch.testing.assert_close(output, input)


class AotKernelTest(unittest.TestCase):
    def test_unique_naming(self):
        # We test it twice to ensure that local name collisions cannot happen,
        # verifying that each run generates a uniquely named kernel. This is
        # a by-product of the Torch namespace being global and every one of
        # these that we define being a separate incarnation based on the
        # same local function name.
        unique_names = set()
        for _ in range(2):
            exported = export_softmax_kernel()
            exported.print_readable()
            ir_text = str(exported.mlir_module)
            matches = re.findall(
                r"flow.dispatch @(tk_kernel_softmax__([0-9]+))::", ir_text
            )
            self.assertEqual(1, len(matches))
            match = matches[0]
            print("NAME MATCH:", match)
            self.assertNotIn(match, unique_names)
            unique_names.add(match)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
