# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import unittest

import torch
from maraboupy import Marabou
from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX

from fossil.domains import inf_bounds_n
from fossil.consts import ActivationType
from fossil.shared.components.estimation import EstimNet


class DiffNetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dimension = 2
        # V - self.f needs to be an EstimNet
        self.f = EstimNet(self.dimension, 1, self.dimension)
        # self.f.fc1.weight = torch.nn.Parameter(torch.ones_like(self.f.fc1.weight))
        # self.f.fc2.weight = torch.nn.Parameter(torch.ones_like(self.f.fc2.weight))
        self.V = NNDiscrete(
            self.dimension,
            None,
            *[1],
            bias=False,
            activate=[ActivationType.RELU],
            llo=False
        )
        # self.V.layers[0].weight = torch.nn.Parameter(torch.ones_like(self.V.layers[0].weight))
        self.DV = _DiffNet(self.V, self.f)

    def test_onnx_export(self):
        translator = MarabouTranslator(self.dimension)
        res = translator.get(net=self.V, ENet=self.f)
        V = res["V"]
        dV = res["V_dot"]
        self.assertIsInstance(V, Marabou.MarabouNetworkONNX)
        self.assertIsInstance(dV, MarabouNetworkONNX)


class MarabouTranslatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dimension = 2
        self.certificate = LyapunovCertificate([None])
        self.f = EstimNet(self.dimension, 1, self.dimension)
        self.learner = NNDiscrete(
            2, self.certificate.learn, *[10], bias=False, activate=[ActivationType.RELU]
        )
        self.translation = None

    def test_Translation(self):
        translator = MarabouTranslator(self.dimension)
        res = translator.get(net=self.learner, ENet=self.f)
        V = res["V"]
        dV = res["V_dot"]
        self.assertIsInstance(V, Marabou.MarabouNetworkONNX)
        self.assertIsInstance(dV, Marabou.MarabouNetworkONNX)

    def test_pass_to_verifier(self):
        # This just checks that the call to self.verify doesn't fail, not that the verication is correct.
        translator = MarabouTranslator(self.dimension)
        res = translator.get(net=self.learner, ENet=self.f)
        verifier = MarabouVerifier(2, None, inf_bounds_n(2), range(2), XD=100)
        V, dV = res["V"], res["V_dot"]
        res = verifier.verify(V, dV)
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    torch.manual_seed(167)
    unittest.main()
