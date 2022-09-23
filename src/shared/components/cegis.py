# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch

from experiments.benchmarks.models import ClosedLoopModel, GeneralClosedLoopModel
import src.certificate as certificate
import src.learner as learner
from src.shared.cegis_values import CegisComponentsState, CegisConfig, CegisStateKeys
from src.shared.components.consolidator import Consolidator
from src.shared.consts import (
    ConsolidatorType,
    LearnerType,
    VerifierType,
    CertificateType,
)
from src.shared.utils import print_section, vprint
import src.translator as translator
import src.verifier as verifier


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, **kw):
        self.n = kw[CegisConfig.N_VARS.k]
        # control layers
        self.ctrl = kw[CegisConfig.CTRLAYER.k]
        # components type
        self.verifier_type = kw[CegisConfig.VERIFIER.k]
        self.certificate_type = kw.get(CegisConfig.CERTIFICATE.k)
        self.time_domain = kw.get(CegisConfig.TIME_DOMAIN.k, CegisConfig.TIME_DOMAIN.v)
        self.consolidator_type = kw.get(
            CegisConfig.CONSOLIDATOR.k, CegisConfig.CONSOLIDATOR.v
        )
        self.time_domain = kw.get(CegisConfig.TIME_DOMAIN.k, CegisConfig.TIME_DOMAIN.v)
        self.learner_type = learner.get_learner(self.time_domain, self.ctrl)
        self.translator_type = translator.get_translator_type(
            self.time_domain, self.verifier_type
        )
        # benchmark opts
        self.h = kw[CegisConfig.N_HIDDEN_NEURONS.k]
        self.activations = kw[CegisConfig.ACTIVATION.k]
        self.system = kw[CegisConfig.SYSTEM.k]
        self.fcts = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.eq = kw.get(
            CegisConfig.EQUILIBRIUM.k, CegisConfig.EQUILIBRIUM.v[0](self.n)
        )
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.rounding = kw.get(CegisConfig.ROUNDING.k, CegisConfig.ROUNDING.v)
        self.ENet = kw.get(CegisConfig.ENET.k, CegisConfig.ENET.v)
        # other opts
        self.max_cegis_iter = kw.get(
            CegisConfig.CEGIS_MAX_ITERS.k, CegisConfig.CEGIS_MAX_ITERS.v
        )
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        # batch init
        self.learning_rate = kw.get(
            CegisConfig.LEARNING_RATE.k, CegisConfig.LEARNING_RATE.v
        )

        # Verifier init
        verifier_type = verifier.get_verifier_type(self.verifier_type)

        self.x = verifier_type.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_domains, self.S, vars_bounds = self.system()

        self.domains = {lab: dom(self.x) for lab, dom in self.f_domains.items()}
        certificate_type = certificate.get_certificate(self.certificate_type)
        self.certificate = certificate_type(domains=self.domains, **kw)

        self.verifier = verifier.get_verifier(
            verifier_type,
            self.n,
            self.certificate.get_constraints,
            vars_bounds,
            self.x,
            **kw
        )

        self.xdot = self.f(self.x)

        # Learner init
        self.learner = self.learner_type(
            self.n,
            self.certificate.learn,
            *self.h,
            bias=self.certificate.bias,
            activate=self.activations,
            equilibria=self.eq,
            llo=self.llo,
            **kw
        )

        self.optimizer = torch.optim.AdamW(
            self.learner.parameters(), lr=self.learning_rate
        )

        if self.consolidator_type == ConsolidatorType.DEFAULT:
            self.consolidator = Consolidator(self.f)
        else:
            TypeError("Not Implemented Consolidator")

        # Translator init
        self.translator = translator.get_translator(
            self.translator_type,
            self.learner,
            self.x,
            self.xdot,
            self.eq,
            self.rounding,
            **kw
        )
        self._result = None

    def solve(self):

        Sdot = {lab: self.f(S) for lab, S in self.S.items()}
        S = self.S

        # the CEGIS loop
        iters = 0
        stop = False

        components = [
            {
                CegisComponentsState.name: "learner",
                CegisComponentsState.instance: self.learner,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "translator",
                CegisComponentsState.instance: self.translator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "verifier",
                CegisComponentsState.instance: self.verifier,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "consolidator",
                CegisComponentsState.instance: self.consolidator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
        ]

        state = {
            CegisStateKeys.net: self.learner,
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.factors: self.fcts,
            CegisStateKeys.V: None,
            CegisStateKeys.V_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.xdot: self.xdot,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None,
            CegisStateKeys.ENet: self.ENet,
        }

        # reset timers
        self.learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        while not stop:
            for component_idx in range(len(components)):
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                if self.verbose:
                    print_section(component[CegisComponentsState.name], iters)
                outputs = component[CegisComponentsState.instance].get(**state)

                state = {**state, **outputs}

                state = {
                    **state,
                    **(
                        component[CegisComponentsState.to_next_component](
                            outputs,
                            next_component[CegisComponentsState.instance],
                            **state
                        )
                    ),
                }

                if state[CegisStateKeys.found] and component_idx == len(components) - 1:
                    if self.certificate_type == CertificateType.RSWS:
                        stop = self.certificate.stay_in_goal_check(
                            self.verifier,
                            state[CegisStateKeys.V],
                            state[CegisStateKeys.V_dot],
                        )
                        if stop:
                            print(f"Found a valid {self.certificate_type.name} certificate")
                    else:
                        print(f"Found a valid {self.certificate_type.name} certificate")
                        stop = True

                if state[CegisStateKeys.verification_timed_out]:
                    print("Verification Timed Out")
                    stop = True

            if self.max_cegis_iter == iters and not state[CegisStateKeys.found]:
                print("Out of Cegis loops")
                stop = True

            iters += 1
            if not (
                state[CegisStateKeys.found]
                or state[CegisStateKeys.verification_timed_out]
            ):
                if state[CegisStateKeys.trajectory] != []:
                    lie_label = [key for key in S.keys() if "lie" in key][0]
                    state[CegisStateKeys.cex][lie_label] = torch.cat(
                        [
                            state[CegisStateKeys.cex][lie_label],
                            state[CegisStateKeys.trajectory],
                        ]
                    )
                (
                    state[CegisStateKeys.S],
                    state[CegisStateKeys.S_dot],
                ) = self.add_ces_to_data(
                    state[CegisStateKeys.S],
                    state[CegisStateKeys.S_dot],
                    state[CegisStateKeys.cex],
                )
                if isinstance(self.f, ClosedLoopModel) or isinstance(self.f, GeneralClosedLoopModel):
                    # It might be better to have a CONTROLLED param to cegis, but there's
                    # already a lot of those so tried to avoid that.
                    optim = torch.optim.AdamW(self.f.controller.parameters())
                    self.f.controller.learn(
                        state[CegisStateKeys.S][self.certificate.XD],
                        self.f.open_loop,
                        optim,
                    )
                    state.update({CegisStateKeys.xdot: self.f(self.x)})

        state[CegisStateKeys.components_times] = [
            self.learner.get_timer().sum,
            self.translator.get_timer().sum,
            self.verifier.get_timer().sum,
            self.consolidator.get_timer().sum,
        ]
        vprint(["Learner times: {}".format(self.learner.get_timer())], self.verbose)
        vprint(
            ["Translator times: {}".format(self.translator.get_timer())], self.verbose
        )
        vprint(["Verifier times: {}".format(self.verifier.get_timer())], self.verbose)
        vprint(
            ["Consolidator times: {}".format(self.consolidator.get_timer())],
            self.verbose,
        )

        self._result = state, np.array(self.x).reshape(-1, 1), self.f, iters
        return self._result

    @property
    def result(self):
        return self._result

    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S: torch tensor
        :param Sdot: torch tensor
        :param ces: list of ctx
        :return:
                S: torch tensor, added new ctx
                Sdot torch tensor, added  f(new_ctx)
        """
        for lab, cex in ces.items():
            if cex != []:
                S[lab] = torch.cat([S[lab], cex], dim=0).detach()
                Sdot[lab] = self.f(S[lab])
        return S, Sdot

    def _assert_state(self):
        assert self.verifier_type in [
            VerifierType.Z3,
            VerifierType.DREAL,
            VerifierType.MARABOU,
        ]
        assert self.learner_type in [LearnerType.CONTINUOUS, LearnerType.DISCRETE]
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.max_cegis_time > 0


class RASCegis:
    """Convenience class for ReachAvoidStay Synthesis
    
    This class is a wrapper for the Cegis class. It is used to run the Cegis algorithm twice,
    once for a Lyapunov function (stability) and once for a Barrier function (safety).
    
    A reach avoid stay criterion relies on an open set D, compact sets XI, XG and a closed set XU.
    http://arxiv.org/abs/2009.04432, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9483376.

    Necessarily there exists A \subset G. Goal is to synth two smooth functions V, B such that:

    (1) V is positive definite wrt A (V(x) = 0 iff x \in A)
    (2) \forall x in D \ A: dV/dt < 0
    (3) \forall x \in XI, B(x) >= 0; \forall x in XU: B(x) <0
    (4) \forall x \in D: dB/dt >= 0"""

    def __init__(self, lyap, barr):
        """_summary_

        Args:
            lyap (dict): dictionary of options for Cegis for Lyapunov synthesis
            barr (dict): dictionary of options for Cegis for Barrier synthesis
        """
        self.c_lyap = Cegis(**lyap)
        self.c_barr = Cegis(**barr)

    def solve(self):
        res_lyap = self.c_lyap.solve()
        res_barr = self.c_barr.solve()
        return res_lyap, res_barr
