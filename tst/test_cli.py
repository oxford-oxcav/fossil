import unittest
from unittest.mock import patch, mock_open
import argparse
import os
import yaml

from fossil import consts
from fossil import cli
from fossil.main import synthesise


class TestCLIHelpers(unittest.TestCase):
    def test_string_to_learning_factors(self):
        s = "NONE"
        result = cli.CegisConfigParser.string_to_enum_value(s, consts.LearningFactors)
        self.assertEqual(result, consts.LearningFactors.NONE)

    def test_string_to_time_domain(self):
        s = "CONTINUOUS"
        result = cli.CegisConfigParser.string_to_enum_value(s, consts.TimeDomain)
        self.assertEqual(result, consts.TimeDomain.CONTINUOUS)

    def test_string_to_verifier_type(self):
        s = "Z3"
        result = cli.CegisConfigParser.string_to_enum_value(s, consts.VerifierType)
        self.assertEqual(result, consts.VerifierType.Z3)

    def test_read_certificate_type(self):
        s = "BARRIER"
        result = cli.CegisConfigParser.string_to_enum_value(s, consts.CertificateType)
        self.assertEqual(result, consts.CertificateType.BARRIER)

    def test_string_to_activation_type(self):
        s = "SIGMOID"
        result = cli.CegisConfigParser.string_to_enum_value(s, consts.ActivationType)
        self.assertEqual(result, consts.ActivationType.SIGMOID)

    def test_read_activations(self):
        s = ["SIGMOID", "TANH"]
        result = cli.CegisConfigParser.read_activations(s, consts.ActivationType)
        self.assertEqual(
            result, [consts.ActivationType.SIGMOID, consts.ActivationType.TANH]
        )

    def test_read_domains(self):
        s = {"XD": "Rectangle([0, 0], [1,1])", "XI": "Sphere([1.0, 0.0], 1.0)"}
        result = cli.CegisConfigParser.read_domains(s)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.keys(), {"lie", "init"})

    def test_read_domains_invalid(self):
        s = {"XD": "Rectangle([0, 0], [1,1])", "XI": "Sphere([1.0, 0.0], 1.0)"}
        with self.assertRaises(Exception):
            cli.CegisConfigParser.read_domains("invalid")


class TestParseYamlToCegisConfig(unittest.TestCase):
    def test_parse_yaml_to_cegis_config_missing_field(self):
        # Define a minimal configuration
        yaml_data = """
        SYSTEM: "system data"
        CERTIFICATE: "certificate data"
        DOMAINS: "domains data"
        """
        # Mock the file reading and yaml loading operations
        with patch(
            "builtins.open", mock_open(read_data=yaml_data)
        ) as mock_file, patch.object(
            yaml, "safe_load", return_value=yaml.load(yaml_data, Loader=yaml.SafeLoader)
        ):
            # This should raise an exception since TIME_DOMAIN is missing
            with self.assertRaises(ValueError):
                cli.parse_yaml_to_cegis_config("dummy_path.yaml")

    def test_parse_yaml_to_cegis_config_missing_field(self):
        # Define a minimal configuration
        yaml_data = """
        SYSTEM: "system data"
        CERTIFICATE: "certificate data"
        DOMAINS: "domains data"
        """
        # Mock the file reading and yaml loading operations
        with patch(
            "builtins.open", mock_open(read_data=yaml_data)
        ) as mock_file, patch.object(
            yaml, "safe_load", return_value=yaml.load(yaml_data, Loader=yaml.SafeLoader)
        ):
            # This should raise an exception since TIME_DOMAIN is missing
            with self.assertRaises(ValueError):
                cli.parse_yaml_to_cegis_config("dummy_path.yaml")


class TestValidYamlFile(unittest.TestCase):
    @patch("os.path.exists", return_value=False)
    def test_invalid_path(self, mock_exists):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.valid_yaml_file("invalid/path.yaml")

    def test_invalid_extension(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.valid_yaml_file("path/to/file.txt")

    @patch("os.path.exists", return_value=True)
    def test_valid_file(self, mock_exists):
        result = cli.valid_yaml_file("path/to/file.yaml")
        self.assertEqual(result, "path/to/file.yaml")


class TestRunCegisFromYaml(unittest.TestCase):

    def test_all_yaml_files(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = '/experiments/benchmarks/cli/'
        onlyfiles = [f for f in os.listdir(parent_dir+file_path)
                     if os.path.isfile(os.path.join(parent_dir+file_path, f))]

        for f in onlyfiles:
            cc = cli.parse_yaml_to_cegis_config(parent_dir + file_path + f)
            c = synthesise(cc)


if __name__ == "__main__":
    unittest.main()
