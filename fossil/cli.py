# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import warnings
from typing import Union, Callable
import argparse
import os
import sys

import yaml

from fossil import consts
from fossil import logger
from fossil import parser
from fossil import control
from fossil import domains as doms


class CegisConfigParser:
    @staticmethod
    def string_to_enum_value(
        value: str, enum_type
    ) -> Union[
        consts.LearningFactors,
        consts.TimeDomain,
        consts.VerifierType,
        consts.CertificateType,
        consts.ActivationType,
    ]:
        """Helper function to convert a string to its corresponding Enum value."""
        try:
            return enum_type[value.upper()]
        except KeyError:
            raise ValueError(f"'{value}' is not a valid {enum_type.__name__}.")

    @staticmethod
    def read_activations(activations: list[str], _) -> list[consts.ActivationType]:
        return [
            CegisConfigParser.string_to_enum_value(activation, consts.ActivationType)
            for activation in activations
        ]

    @staticmethod
    def read_model(
        dynamics: list[str], verifier: consts.VerifierType, controller_layer
    ) -> Callable:
        """Read model based on presence of a controller layer."""
        model_cls = (
            control._ParsedDynamicalModel
            if controller_layer is None
            else control._ParsedControllableDynamicalModel
        )
        model = model_cls(dynamics, verifier)

        if controller_layer is None:
            return lambda: model

        return control.GeneralClosedLoopModel.prepare_from_open(model)

    @staticmethod
    def get_domains_from_dict(domains_dict: dict[str, str]) -> dict:
        names = consts.DomainNames
        return {
            names[key].value: parser.parse_domain(value)
            for key, value in domains_dict.items()
        }

    @staticmethod
    def read_model(
        dynamics: list[str], verifier: consts.VerifierType, controller_layer
    ) -> Callable:
        """Read model based on presence of a controller layer."""
        model_cls = (
            control._ParsedDynamicalModel
            if controller_layer is None
            else control._ParsedControllableDynamicalModel
        )
        model = model_cls(dynamics, verifier)

        if controller_layer is None:
            return lambda: model

        return control.GeneralClosedLoopModel.prepare_from_open(model)

    @staticmethod
    def read_domains(domains: dict[str, str]) -> dict:
        # key value pairs for domain names and their corresponding Domain objects
        names = consts.DomainNames
        return {
            names[key].value: parser.parse_domain(value)
            for key, value in domains.items()
        }

    @staticmethod
    def add_borders(domains: dict, certificate: consts.CertificateType):
        borders = certificate.get_required_borders(certificate)
        for domain, border in borders.items():
            # The true domain is stored, and then cegis handles generating the boundary
            # of the domain symbolically
            domains[border.value] = domains[domain.value]
        return domains

    @staticmethod
    def read_data(data: Union[dict[str, str], float], domains):
        # key value pairs for domain names and their corresponding Domain objects
        borders = consts.DomainNames.border_sets().values()
        data_dict = {}
        if isinstance(data, (float, int)):
            for key in domains.keys():
                if key in borders:
                    data_dict[key] = domains[key]._sample_border(data)
                else:
                    data_dict[key] = domains[key]._generate_data(data)
            return data_dict
        else:
            DN = consts.DomainNames
            for key, N in data.items():
                domain = DN[key]
                data_dom = domain.value
                if domain in borders:
                    data_dict[data_dom] = domains[data_dom]._sample_border(N)
                elif domain.value in domains.keys():
                    data_dict[data_dom] = domains[data_dom]._generate_data(N)
            return data_dict

    @staticmethod
    def adjust_roa(data, domains, data_config):
        XD = consts.DomainNames.XD.value
        domains.pop(XD)
        return data, domains

    @staticmethod
    def adjust_rwa(data, domains, data_config):
        XD = consts.DomainNames.XD.value
        XS = consts.DomainNames.XS.value
        XU = consts.DomainNames.XU.name
        XU_val = consts.DomainNames.XU.value
        unsafe = doms.SetMinus(domains[XD], domains[XS])
        N_XU = data_config[XU] if isinstance(data_config, dict) else data_config
        data[XU_val] = unsafe._generate_data(N_XU)

        return data, domains

    @staticmethod
    def adjust_rswa(data, domains, data_config):
        XD = consts.DomainNames.XD.value
        XS = consts.DomainNames.XS.value
        XU = consts.DomainNames.XU.name
        XU_val = consts.DomainNames.XU.value
        unsafe = doms.SetMinus(domains[XD], domains[XS])
        N_XU = data_config[XU] if isinstance(data_config, dict) else data_config
        data[XU_val] = unsafe._generate_data(N_XU)
        return data, domains

    @staticmethod
    def adjust_rar(data, domains, data_config):
        XD = consts.DomainNames.XD.value
        XS = consts.DomainNames.XS.value
        XU = consts.DomainNames.XU.name
        XU_val = consts.DomainNames.XU.value
        unsafe = doms.SetMinus(domains[XD], domains[XS])
        XF = consts.DomainNames.XF.value
        XNF = doms.SetMinus(domains[XD], domains[XF])
        N_XU = data_config[XU] if isinstance(data_config, dict) else data_config
        data[XU_val] = unsafe._generate_data(N_XU)
        N_XNF = (
            data_config[consts.DomainNames.XNF.name]
            if isinstance(data_config, dict)
            else data_config
        )
        data[consts.DomainNames.XNF.value] = XNF._generate_data(N_XNF)

        return data, domains

    @staticmethod
    def adjust_data_and_domains_for_certificate(
        data, domains, certificate, data_config
    ):
        if certificate == consts.CertificateType.ROA:
            return CegisConfigParser.adjust_roa(data, domains, data_config)
        elif certificate in (consts.CertificateType.RWS, consts.CertificateType.RWA):
            return CegisConfigParser.adjust_rwa(data, domains, data_config)
        elif certificate in (consts.CertificateType.RSWS, consts.CertificateType.RSWA):
            return CegisConfigParser.adjust_rswa(data, domains, data_config)
        elif certificate == consts.CertificateType.RAR:
            return CegisConfigParser.adjust_rar(data, domains, data_config)

        return data, domains


def parse_yaml_to_cegis_config(file_path: str) -> consts.CegisConfig:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Ensure required fields are present
    for required_field in ["SYSTEM", "CERTIFICATE", "DOMAINS", "TIME_DOMAIN"]:
        if required_field not in config_data:
            raise ValueError(
                f"Required field {required_field} is missing from the YAML file!"
            )

    # Populate the CegisConfig object
    config = consts.CegisConfig()
    CFP = CegisConfigParser

    for field, value in config_data.items():
        if hasattr(config, field):
            if field in ENUM_ATTR_MAP:
                enum_type = enum_type_map[field]
                setattr(
                    config,
                    field,
                    ENUM_ATTR_MAP[field](value, enum_type),
                )
            else:
                setattr(config, field, value)
        else:
            warnings.warn(f"Warning: Unknown field {field} in the YAML file. Ignoring.")

    system = CFP.read_model(config_data["SYSTEM"], config.VERIFIER, config.CTRLAYER)
    domains = CFP.read_domains(config_data["DOMAINS"])
    domains = CFP.add_borders(domains, config.CERTIFICATE)
    data = CFP.read_data(config_data["N_DATA"], domains)
    data, domains = CFP.adjust_data_and_domains_for_certificate(
        data, domains, config.CERTIFICATE, config_data["N_DATA"]
    )
    config.SYSTEM = system
    config.DATA = data
    config.DOMAINS = domains

    return config


ENUM_ATTR_MAP = {
    "CERTIFICATE": CegisConfigParser.string_to_enum_value,
    "TIME_DOMAIN": CegisConfigParser.string_to_enum_value,
    "VERIFIER": CegisConfigParser.string_to_enum_value,
    "FACTORS": CegisConfigParser.string_to_enum_value,
    "ACTIVATION": CegisConfigParser.read_activations,
    "CTRLACTIVATION": CegisConfigParser.read_activations,
    "ACTIVATION_ALT": CegisConfigParser.read_activations,
}

enum_type_map = {
    "CERTIFICATE": consts.CertificateType,
    "TIME_DOMAIN": consts.TimeDomain,
    "VERIFIER": consts.VerifierType,
    "FACTORS": consts.LearningFactors,
    "ACTIVATION": consts.ActivationType,
    "CTRLACTIVATION": consts.ActivationType,
    "ACTIVATION_ALT": consts.ActivationType,
}


def valid_yaml_file(filename):
    if not os.path.exists(filename):
        raise argparse.ArgumentTypeError(f"The file '{filename}' does not exist.")
    if not filename.endswith(".yaml"):
        raise argparse.ArgumentTypeError(f"The file '{filename}' is not a .yaml file.")
    return filename


def print_certificate_sets(certificate: str):
    certificate = CegisConfigParser.string_to_enum_value(
        certificate.upper(), consts.CertificateType
    )
    domains, data = consts.CertificateType.get_certificate_sets(certificate)

    # Display certificate type
    print("\033[1mCertificate Type:\033[0m")
    print(f"{certificate.name} (Usage)")

    # Divider
    print("-" * 40)

    # Display required symbolic domains
    print("\033[1mRequired Symbolic Domains:\033[0m")
    for domain in domains:
        print(f"  - {domain.name} ({domain.value})")

    # Divider
    print()

    # Display required data domains
    print("\033[1mRequired Data Domains:\033[0m")
    for domain in data:
        print(f"  - {domain.name} ({domain.value})")

    # Divider
    print("-" * 40)

    print(
        "Note: Symbolic Border domains do not need to be explicitly specified. "
        "They are automatically generated, though the number of data points may still be specified."
        "Similarly, data domains that are not explicitly specified in the domains field are automatically generated."
        "For more information, see the documentation."
    )

    sys.exit()


def parse_filename():
    parser = argparse.ArgumentParser(description="Fossil CLI")
    parser.add_argument(
        "file",
        type=valid_yaml_file,
        help="Path to the YAML configuration file.",
        nargs="?",  # Indicates the argument is optional
        default=None,  # Default value when the argument is not provided
    )

    parser.add_argument(
        "--certificate",
        type=str.upper,
        help="Pass a certificate to view what sets are required for synthesis.",
        choices=[cert.name for cert in consts.CertificateType],
        default=None,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help="Set verbosity level. 0 is silent (default), 1 is verbose, 2 is debug.",
        choices=[0, 1, 2],
        default=0,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the resulting certificate (2D-only).",
    )

    args = parser.parse_args()
    logger.Logger.set_logger_level(args.verbose)

    if args.file is None and args.certificate is None:
        parser.error("Either 'file' or '--certificate' must be provided.")

    return args
