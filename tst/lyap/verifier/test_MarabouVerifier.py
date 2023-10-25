# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import unittest
import tempfile

import torch
from maraboupy import Marabou

from fossil.domains import inf_bounds_n
import fossil.verifier as verifier


class ReluNet(torch.nn.Module):
    def __init__(self, c: float) -> None:
        super(ReluNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1, bias=False),
        )
        with torch.no_grad():
            self.model[0].weight = torch.nn.Parameter(torch.tensor([[1.0], [-1.0]]))
            self.model[2].weight = torch.nn.Parameter(
                c * torch.ones_like(self.model[2].weight)
            )

    def forward(self, x):
        return self.model(x)


class MarabouVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        net_pos = ReluNet(2.0)
        net_neg = ReluNet(-1.0)
        self.net_neg_file = tempfile.NamedTemporaryFile(suffix=".onnx")
        self.net_pos_file = tempfile.NamedTemporaryFile(suffix=".onnx")
        dummy_input = torch.rand([1, 1])
        torch.onnx.export(
            net_pos,
            dummy_input,
            self.net_pos_file,
            input_names=["S", "Sdot"],
            output_names=["V"],
        )
        torch.onnx.export(
            net_neg,
            dummy_input,
            self.net_neg_file,
            input_names=["S", "Sdot"],
            output_names=["V"],
        )
        return super().setUp()

    def test_Verification_unsat(self):
        ver = verifier.VerifierMarabou(
            1, None, inf_bounds_n(1), range(1), INNER_RADIUS=0.1
        )
        V = Marabou.read_onnx(
            self.net_pos_file.name,
        )
        Vdot = Marabou.read_onnx(self.net_neg_file.name)
        res = ver.verify(V, Vdot)
        self.assertTrue(res["found"])

    def test_Verification_sat(self):
        ver = verifier.VerifierMarabou(
            1, None, inf_bounds_n(1), range(1), INNER_RADIUS=0.1
        )
        V = Marabou.read_onnx(self.net_neg_file.name)
        Vdot = Marabou.read_onnx(self.net_pos_file.name)
        res = ver.verify(V, Vdot)
        self.assertFalse(res["found"])


if __name__ == "__main__":
    torch.manual_seed(167)
    unittest.main()
