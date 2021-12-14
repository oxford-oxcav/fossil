# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import tempfile
from typing import Tuple, Union

import torch
from maraboupy.Marabou import read_onnx
from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX

from src.shared.component import Component
from src.learner.net_discrete import NNDiscrete
from src.shared.utils import Timer, timer
from src.shared.cegis_values import CegisStateKeys

T = Timer()


class _DiffNet(torch.nn.Module):
    """ Private class to provide forward method of delta_V for Marabou

    V (NNDiscrete): Candidate Lyapunov ReluNet
    f (EstimNet): Estimate of system dynamics ReluNet
    """

    def __init__(self, V: NNDiscrete, f) -> None:
        super(_DiffNet, self).__init__()
        self.V = V
        # Means forward can only be called with batchsize = 1
        self.factor = torch.nn.Parameter(-1 * torch.ones([1, 1]))
        self.F = f

    def forward(self, S, Sdot) -> torch.Tensor:
        return self.V(self.F(S), self.F(Sdot))[0] + self.factor @ self.V(S, Sdot)[0]


class MarabouTranslator(Component):
    """ Takes an torch nn.module object and converts it to an onnx file to be read by marabou

    dimension (int): Dimension of dynamical system
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    @timer(T)
    def get(self, net: NNDiscrete = None, ENet=None, **kw) -> Tuple[MarabouNetworkONNX, MarabouNetworkONNX]:
        """
        net (NNDiscrete): PyTorch candidate Lyapunov Neural Network
        ENet (EstimNet): dynamical system as PyTorch Neural Network
        """
        tf_V = tempfile.NamedTemporaryFile(suffix='.onnx')
        tf_DV = tempfile.NamedTemporaryFile(suffix='.onnx')
        model = _DiffNet(net, ENet)
        self.export_net_to_file(net, tf_V, "V")
        self.export_net_to_file(model, tf_DV, "dV")
        
        V_net = read_onnx(tf_V.name, outputName='V')                  
        dV_net = read_onnx(tf_DV.name, outputName='dV')
        return {CegisStateKeys.V: V_net, CegisStateKeys.V_dot: dV_net}

    def export_net_to_file(self, net: Union[_DiffNet, NNDiscrete],  tf,  output: str) -> None:
        dummy_input = (torch.rand([1, self.dimension]), torch.rand([1, self.dimension]))
        torch.onnx.export(net, dummy_input, tf, input_names=['S', 'Sdot'],
                          output_names=[output], opset_version=11)

    @staticmethod
    def get_timer():
        return T
