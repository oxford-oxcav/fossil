# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import replace
from itertools import chain
from typing import Union, NamedTuple

import torch

import fossil.logger as logger
import fossil.certificate as certificate
import fossil.consolidator as consolidator
from fossil.consts import Any, torch
import fossil.control as control
import fossil.learner as learner
import fossil.translator as translator
import fossil.verifier as verifier
from fossil.consts import *

cegis_log = logger.Logger.setup_logger(__name__)


class CegisStats(NamedTuple):
    iters: int
    N_data: int
    times: dict
    seed: int


class Result(NamedTuple):
    res: bool
    cert: learner.LearnerNN
    f: Any
    cegis_stats: CegisStats


class SingleCegis:
    def __init__(self, config: CegisConfig):
        self.config = config
        self.x, self.x_map, self.domains = self._initialise_domains()
        self.f, self.xdot = self._initialise_system()
        self.S = self._initialise_data()
        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        self.verifier = self._initialise_verifier()
        self.optimizer = self._initialise_optimizer()
        self.consolidator = self._initialise_consolidator()
        self.translator_type, self.translator = self._initialise_translator()
        self._result = None
        assert self.x is self.verifier.xs
        self._assert_state()
        if self.config.VERBOSE:
            logger.Logger.set_logger_level(self.config.VERBOSE)

    def _initialise_learner(self):
        learner_type = learner.get_learner(
            self.config.TIME_DOMAIN, self.config.CTRLAYER
        )
        learner_instance = learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            bias=self.certificate.bias,
            config=self.config,
        )
        return learner_instance

    def _initialise_verifier(self):
        verifier_type = verifier.get_verifier_type(self.config.VERIFIER)
        verifier_instance = verifier.get_verifier(
            verifier_type,
            self.config.N_VARS,
            self.certificate.get_constraints,
            self.x,
            self.config.VERBOSE,
        )
        return verifier_instance

    def _initialise_system(self):
        system = self.config.SYSTEM
        ctrler = None
        f = None
        if self.config.CTRLAYER:
            ctrl_activ = self.config.CTRLACTIVATION
            ctrler = control.GeneralController(
                inputs=self.config.N_VARS,
                output=self.config.CTRLAYER[-1],
                layers=self.config.CTRLAYER[:-1],
                activations=ctrl_activ,
            )
            f = system(ctrler)
        else:
            f = system()
        xdot = f(self.x)
        self.config = replace(self.config, SYSTEM=f)
        return f, xdot

    def _initialise_domains(self):
        x = verifier.get_verifier_type(self.config.VERIFIER).new_vars(
            self.config.N_VARS
        )
        x_map = {str(x): x for x in x}
        domains = {
            label: domain.generate_boundary(x)
            if label in certificate.BORDERS
            else domain.generate_domain(x)
            for label, domain in self.config.DOMAINS.items()
        }
        if self.config.CERTIFICATE == CertificateType.RAR:
            domains[certificate.XNF] = self.config.DOMAINS[
                certificate.XF
            ].generate_complement(x)

        cegis_log.debug("Domains: {}".format(domains))
        return x, x_map, domains

    def _initialise_data(self):
        cegis_log.debug("Data: {}".format(self.config.DATA))
        return {key: S() for key, S in self.config.DATA.items()}

    def _initialise_certificate(self):
        custom_certificate = self.config.CUSTOM_CERTIFICATE
        certificate_type = certificate.get_certificate(self.config.CERTIFICATE, custom_certificate)
        if self.config.CERTIFICATE == certificate.CertificateType.STABLESAFE:
            raise ValueError("StableSafe not compatible with default CEGIS")
        return certificate_type(self.domains, self.config)

    def _initialise_optimizer(self):
        return torch.optim.AdamW(
            [{"params": self.learner.parameters()}, {"params": self.f.parameters()}],
            lr=self.config.LEARNING_RATE,
        )

    def _initialise_consolidator(self):
        if self.config.CONSOLIDATOR == ConsolidatorType.DEFAULT:
            return consolidator.Consolidator(self.f)
        else:
            raise TypeError("Not Implemented Consolidator")

    def _initialise_translator(self):
        translator_type = translator.get_translator_type(
            self.config.TIME_DOMAIN, self.config.VERIFIER
        )
        return translator_type, translator.get_translator(
            translator_type,
            self.x,
            self.xdot,
            self.config.ROUNDING,
            config=self.config,
        )

    def solve(self) -> Result:
        Sdot = {lab: self.f(S) for lab, S in self.S.items()}
        S = self.S

        # Initialize CEGIS state
        state = self.init_state(Sdot, S)

        # Reset timers for components
        self.learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        iters = 0
        stop = False

        while not stop:
            # Learner component
            cegis_log.debug("\033[1m Learner \033[0m")
            outputs = self.learner.get(**state)
            state = {**state, **outputs}

            # Update xdot with new controller if necessary
            state = self.update_controller(state)

            # Translator component
            cegis_log.debug("\033[1m Translator \033[0m")
            outputs = self.translator.get(**state)
            state = {**state, **outputs}

            # Verifier component
            cegis_log.debug("\033[1m Verifier \033[0m")
            outputs = self.verifier.get(**state)
            state = {**state, **outputs}

            # Consolidator component
            cegis_log.debug("\033[1m Consolidator \033[0m")
            outputs = self.consolidator.get(**state)
            state = {**state, **outputs}

            if state[CegisStateKeys.found]:
                stop = self.process_certificate(S, state, iters)

            elif state[CegisStateKeys.verification_timed_out]:
                cegis_log.warning("Verification timed out")
                stop = True

            elif (
                self.config.CEGIS_MAX_ITERS == iters and not state[CegisStateKeys.found]
            ):
                cegis_log.warning("Out of iterations")
                stop = True

            elif not (
                state[CegisStateKeys.found]
                or state[CegisStateKeys.verification_timed_out]
            ):
                state = self.process_cex(S, state)

            iters += 1
            cegis_log.info("Iteration: {}".format(iters))

        state = self.process_timers(state)

        N_data = sum([S_i.shape[0] for S_i in state[CegisStateKeys.S].values()])
        stats = CegisStats(
            iters, N_data, state["components_times"], torch.initial_seed()
        )
        self._result = Result(state[CegisStateKeys.found], state["net"], self.f, stats)
        return self._result

    def update_controller(self, state):
        cegis_log.debug("Updating state xdot with controller")
        state.update({CegisStateKeys.xdot: self.f(self.x)})
        return state

    def init_state(self, Sdot, S):
        state = {
            CegisStateKeys.net: self.learner,
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.V: None,
            CegisStateKeys.V_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.xdot: self.xdot,
            CegisStateKeys.xdot_func: self.f._f_torch,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None,
            CegisStateKeys.ENet: self.config.ENET,
        }

        return state

    def process_cex(
        self, S: dict[str, torch.Tensor], state: dict[str, Any]
    ) -> dict[str, Any]:
        if state[CegisStateKeys.trajectory] != []:
            lie_key = certificate.XD
            lie_label = [key for key in S.keys() if lie_key in key][0]
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
        return state

    def process_timers(self, state: dict[str, Any]) -> dict[str, Any]:
        state[CegisStateKeys.components_times] = [
            self.learner.get_timer().sum,
            self.translator.get_timer().sum,
            self.verifier.get_timer().sum,
            self.consolidator.get_timer().sum,
        ]
        cegis_log.info("Learner times: {}".format(self.learner.get_timer()))
        cegis_log.info("Translator times: {}".format(self.translator.get_timer()))
        cegis_log.info("Verifier times: {}".format(self.verifier.get_timer()))
        cegis_log.info("Consolidator times: {}".format(self.consolidator.get_timer()))
        return state

    def process_certificate(
        self, S: dict[str, torch.Tensor], state: dict[str, Any], iters: int
    ) -> bool:
        stop = False
        if (
            self.config.CERTIFICATE == CertificateType.LYAPUNOV
            or self.config.CERTIFICATE == CertificateType.ROA
        ):
            self.learner.beta = self.certificate.estimate_beta(self.learner)

        if self.config.CERTIFICATE == CertificateType.RSWS:
            stay = self.certificate.beta_search(
                self.learner,
                self.verifier,
                state[CegisStateKeys.V],
                state[CegisStateKeys.V_dot],
                S,
            )
            # Only stop if we prove the final stay condition
            stop = stay
            if stay:
                print(f"Found a valid {self.config.CERTIFICATE.name} certificate")
            else:
                print(
                    f"Found a valid RWS certificate, but could not prove the final stay condition. Keep searching..."
                )
                state[CegisStateKeys.found] = False
                if self.config.CEGIS_MAX_ITERS == iters:
                    stop = True
        else:
            if isinstance(self.f, control.GeneralClosedLoopModel):
                ctrl = " and controller"
            else:
                ctrl = ""
            print(f"Found a valid {self.config.CERTIFICATE.name} certificate" + ctrl)
            stop = True
        return stop

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
        assert self.config.LEARNING_RATE > 0
        assert self.config.CEGIS_MAX_TIME_S > 0
        if self.config.TIME_DOMAIN == TimeDomain.DISCRETE:
            assert self.config.CERTIFICATE in (
                CertificateType.LYAPUNOV,
                CertificateType.BARRIERALT,
            )
        # Passing sets to Fossil is complicated atm and I've messed it up (passing too many can lead to bugs too).
        # This is a temporary debug check until some better way of passing sets is implemented.
        self.certificate._assert_state(self.domains, self.S)


class DoubleCegis(SingleCegis):
    """StableSafe Cegis in parallel.

    A stable while stay criterion relies on an open set D, compact sets XI, XG and a closed set XU.
    http://arxiv.org/abs/2009.04432, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9483376.

    Necessarily there exists A \subset G. Goal is to synth two smooth functions V, B such that:

    (1) V is positive definite wrt A (V(x) = 0 iff x \in A)
    (2) \forall x in D \ A: dV/dt < 0
    (3) \forall x \in XI, B(x) >= 0; \forall x in XU: B(x) <0
    (4) \forall x \in D: dB/dt >= 0"""

    def __init__(self, config: CegisConfig):
        super().__init__(config)
        self.lyap_learner, self.barr_learner = self.learner

    def _initialise_certificate(self):
        certificate_type = certificate.get_certificate(self.config.CERTIFICATE)
        if self.config.CERTIFICATE not in (
            CertificateType.STABLESAFE,
            CertificateType.RAR,
        ):
            raise ValueError("DoubleCegis only supports StableSafe certificates")
        return certificate_type(self.domains, self.config)

    def _initialise_learner(self):
        learner_type = learner.get_learner(
            self.config.TIME_DOMAIN, self.config.CTRLAYER
        )

        lyap_learner = learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS,
            bias=self.certificate.bias[0],
            activation=self.config.ACTIVATION,
            config=self.config,
        )

        # When initialising the barrier learner, we want to ensure LLO is False
        # This is an oversight on my part, and should be fixed in the future
        barr_learner = learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS_ALT,
            bias=self.certificate.bias[1],
            activation=self.config.ACTIVATION_ALT,
            config=replace(self.config, LLO=False),
        )

        if self.config.CERTIFICATE == CertificateType.RAR:
            lyap_learner._type = CertificateType.RWS.name
            barr_learner._type = CertificateType.BARRIER.name
        elif self.config.CERTIFICATE == CertificateType.STABLESAFE:
            lyap_learner._type = CertificateType.ROA.name
            barr_learner._type = CertificateType.BARRIER.name

        return lyap_learner, barr_learner

    def _initialise_optimizer(self):
        optimizer = torch.optim.AdamW(
            chain(
                *(l.parameters() for l in self.learner),
                self.f.parameters(),
            ),
            lr=self.config.LEARNING_RATE,
        )
        return optimizer

    def _initialise_translator(self):
        translator_type = translator.TranslatorCTDouble
        translator_instance = translator.get_translator(
            translator_type,
            self.x,
            self.xdot,
            self.config.ROUNDING,
            config=self.config,
        )
        return translator_type, translator_instance

    def solve(self) -> Result:
        Sdot = {lab: self.f(S) for lab, S in self.S.items()}
        S = self.S

        # Initialize CEGIS state
        state = self.init_state(Sdot, S)

        # Reset timers for components
        self.lyap_learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        iters = 0
        stop = False

        while not stop:
            # Learner component
            cegis_log.debug("\033[1m Learner \033[0m")
            outputs = self.lyap_learner.get(**state)
            state = {**state, **outputs}

            # Update xdot with new controller if necessary
            state = self.update_controller(state)

            # Translator component
            cegis_log.debug("\033[1m Translator \033[0m")
            outputs = self.translator.get(**state)
            state = {**state, **outputs}

            # Verifier component
            cegis_log.debug("\033[1m Verifier \033[0m")
            outputs = self.verifier.get(**state)
            state = {**state, **outputs}

            if isinstance(self.certificate, certificate.SafeROA):
                if certificate.XI in state["cex"].keys():
                    # This means we've got the this check in the verifier, so the Lyapunov in SafeROA is correct.
                    if not self.config.CTRLAYER:
                        self.lyap_learner.freeze()
            elif isinstance(self.certificate, certificate.ReachAvoidRemain):
                if certificate.XG in state["cex"].keys():
                    # This means we've got the this check in the verifier, so the RWS function in RAR is correct.
                    if not self.config.CTRLAYER:
                        self.lyap_learner.freeze()

            # Consolidator component does not exist for DoubleCegis
            # if self.config.VERBOSE:
            #     print_section("consolidator", iters)
            # outputs = self.consolidator.get(**state)
            # state = {**state, **outputs}

            if state[CegisStateKeys.found]:
                stop = self.process_certificate(S, state, iters)

            elif state[CegisStateKeys.verification_timed_out]:
                cegis_log.warning("Verification timed out")
                stop = True

            elif (
                self.config.CEGIS_MAX_ITERS == iters and not state[CegisStateKeys.found]
            ):
                cegis_log.warning("Out of iterations")
                stop = True

            elif not (
                state[CegisStateKeys.found]
                or state[CegisStateKeys.verification_timed_out]
            ):
                state = self.process_cex(S, state)

            iters += 1
            cegis_log.info("Iteration: {}".format(iters))

        state = self.process_timers(state)

        N_data = sum([S_i.shape[0] for S_i in state[CegisStateKeys.S].values()])
        stats = CegisStats(
            iters, N_data, state["components_times"], torch.initial_seed()
        )
        self._result = Result(state[CegisStateKeys.found], state["net"], self.f, stats)
        return self._result

    def process_cex(
        self, S: dict[str, torch.Tensor], state: dict[str, Any]
    ) -> dict[str, Any]:
        (
            state[CegisStateKeys.S],
            state[CegisStateKeys.S_dot],
        ) = self.add_ces_to_data(
            state[CegisStateKeys.S],
            state[CegisStateKeys.S_dot],
            state[CegisStateKeys.cex],
        )
        return state

    def process_timers(self, state: dict[str, Any]) -> dict[str, Any]:
        state[CegisStateKeys.components_times] = [
            self.lyap_learner.get_timer().sum,
            self.translator.get_timer().sum,
            self.verifier.get_timer().sum,
            self.consolidator.get_timer().sum,
        ]
        cegis_log.info("Learner times: {}".format(self.lyap_learner.get_timer()))
        cegis_log.info("Translator times: {}".format(self.translator.get_timer()))
        cegis_log.info("Verifier times: {}".format(self.verifier.get_timer()))
        # cegis_log.info("Consolidator times: {}".format(self.consolidator.get_timer()))
        return state


class Cegis:
    def __new__(cls, config: CegisConfig) -> Union[DoubleCegis, SingleCegis]:
        if config.CERTIFICATE in (
            certificate.CertificateType.STABLESAFE,
            certificate.CertificateType.RAR,
        ):
            return DoubleCegis(config)
        else:
            return SingleCegis(config)

    def __init__(self, config: CegisConfig):
        pass

    def solve(self) -> Result:
        raise NotImplementedError("This should be implemented by child classes")
