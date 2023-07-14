# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import csv
import os.path
import warnings
from collections import namedtuple

import pandas as pd


from src.consts import CegisConfig, CertificateType, ACTIVATION_NAMES
from src import certificate


DRF = DEFAULT_RESULT_FILE = os.path.dirname(os.path.realpath(__file__)) + "/results.csv"


@dataclasses.dataclass
class AnalysisConfig:
    """Class for storing configuration for analysis."""

    results_file: str = DRF
    benchmarks: tuple[str] = ()
    latex: bool = True
    latex_file: str = "experiments/main_tab.tex"


Stats = namedtuple("Stats", ["mean", "std", "min", "max"])
BenchmarkData = namedtuple(
    "Benchmark", ["name", "certificate", "success", "times", "loops"]
)

HEADER = [
    "Certificate",
    "N_s",
    "N_u",
    "Result",
    "Benchmark_file",
    "seed",
    "XD",
    "XI",
    "XU",
    "XG",
    "N_Data",
    "Neurons",
    "Activations",
    "Ctrl_Neurons",
    "Ctrl_Activations",
    "Alt_Neurons",
    "Alt_Activations",
    "Factors",
    "LLO",
    "Symmetric_Belt",
    "Time_Domain",
    "Verifier",
    "Loops",
    "Total_Time",
    "Learner_Time",
    "Translator_Time",
    "Verifier_Time",
    "Consolidator_Time",
]


def ratio(x):
    """Returns ratio of True to False values in x."""
    return x.sum() / x.count()


def _get_model_name(model):
    """Returns the class name of the model, or the open_loop class name if model is a GeneralClosedLoopModel."""
    try:
        name = model.__name__
    except AttributeError:
        name = type(model(None).open_loop).__name__
    return name


def get_activations(acts):
    return "[" + ",".join([ACTIVATION_NAMES[act] for act in acts]) + "]"


class CSVWriter:
    """Class for writing results to csv file."""

    def __init__(self, filename: str, headers: list[str]) -> None:
        """Initialises CSVWriter.

        If the file does not exist, it will be created here
        and the header will be written to it.

        Args:
            filename (str): filename of csv file.
            headers (list[str]): headers for csv file
        """
        self.headers = headers
        self.filename = filename
        if not os.path.isfile(self.filename):
            self.write_header_to_csv()

    def write_header_to_csv(self):
        """Creates csv file and writes a header to it."""
        with open(self.filename, "a+") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.headers, delimiter=",", lineterminator="\n"
            )
            writer.writeheader()

    def write_row_to_csv(self, values: list):
        """Writes values to row in CSV file."""
        if len(values) != len(self.headers):
            warnings.warn(
                "Number of values to write ({}) does not match number columns ({}) in csv.".format(
                    len(values), len(self.headers)
                )
            )
        with open(self.filename, "a+") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(values)


class Recorder:
    """
    Class for recording results of experiments.

    The basic assumption is that all experiments will be stored in the same csv file, with as many details as needed.
    Experiments are identified by the filename of the __main__ file, and the seed used to generate the random numbers.
    Crucially, this means that benchmark files should be named uniquely, even across folders,
    and that the seed should be set in the benchmark file.
    Ideally all other config parameters are kept constant for a given benchmark file accross the whole CSV.
    IE if you want to compare different activation functions, we should have a different benchmark file for each one,
    but this is not enforced and might not be the best practice in the end.
    """

    def __init__(self, config=AnalysisConfig) -> None:
        """Initialises ExperimentRecorder.

        Args:
            filename (str): filename of csv file.
        """
        self.filename = config.results_file

    def record(self, config: CegisConfig, result, T: float):
        """records results of abstraction to csv file.

        Args:
            config (_type_): program configuration
            result: cegis return tuple
            T (float): total time taken to run cegis
        """

        headers = HEADER
        model_name = _get_model_name(config.SYSTEM)
        cegis_stats = result.cegis_stats
        timers = cegis_stats.times
        N_data = cegis_stats.N_data
        if config.CTRLAYER is not None:
            N_u = config.CTRLAYER[-1]  # Is this correct?
            ctrlayer = config.CTRLAYER[:-1]
            ctrl_act = [act.name for act in config.CTRLACTIVATION]
        else:
            N_u = 0
            ctrlayer = None
            ctrl_act = None

        if config.CERTIFICATE == CertificateType.STABLESAFE:
            alt_layers = config.N_HIDDEN_NEURONS_ALT
            alt_act = [act.name for act in config.ACTIVATION_ALT]
        else:
            alt_layers = None
            alt_act = None

        XD = config.DOMAINS.get(certificate.XD, None)
        XI = config.DOMAINS.get(certificate.XI, None)
        XU = config.DOMAINS.get(certificate.XU, None)
        XG = config.DOMAINS.get(certificate.XG, None)

        result = [
            config.CERTIFICATE.name,
            config.N_VARS,
            N_u,
            result.res,
            model_name,
            cegis_stats.seed,
            XD,
            XI,
            XU,
            XG,
            N_data,
            config.N_HIDDEN_NEURONS,
            get_activations(config.ACTIVATION),
            ctrlayer,
            ctrl_act,
            alt_layers,
            alt_act,
            config.FACTORS.name,
            config.LLO,
            config.SYMMETRIC_BELT,
            config.TIME_DOMAIN.name,
            config.VERIFIER.name,
            cegis_stats.iters,
            T,
            timers[0],
            timers[1],
            timers[2],
            timers[3],
        ]

        writer = CSVWriter(self.filename, headers)
        writer.write_row_to_csv(result)


class Analyser:
    """Analyse results for given benchmarks."""

    def __init__(self, config=AnalysisConfig()) -> None:
        self.results = pd.read_csv(config.results_file)

    def get_benchmarks(self):
        """Get list of benchmarks in results file."""
        return self.results["Benchmark_file"].unique()

    def analyse_benchmark(self, benchmark_file: str):
        """Analyse results for given benchmark, across all seeds.

        If multiple rows have the same seed, the first one will be kept.
        Does not distinguish between other config options, so these should be kept constant,
        or another function should be used to analyse the results.

        Args:
            filename (str): filename of csv file.
        """

        df = self.results[self.results["Benchmark_file"] == benchmark_file]
        self.check_sanitisation(df)

        df = df.drop(
            columns=[
                "XD",
                "XI",
                "XU",
                "XG",
                "Neurons",
                "Activations",
                "Factors",
                "LLO",
                "Symmetric_Belt",
                "Time_Domain",
                "Verifier",
            ]
        )
        df = df.drop_duplicates(subset=["seed"])
        SR = self.get_success_ratio(df)

        # Only analyse successful runs
        df = df[df["Result"] == True]
        times = self.time_analysis(df)
        loops = self.loop_analysis(df)
        cert = df["Certificate"].unique()[0]
        benchmark = BenchmarkData(benchmark_file, cert, SR, times, loops)
        return benchmark

    @staticmethod
    def check_sanitisation(df):
        """Check that all columns have the same value. Warn if not.
        Assumes frame consists of only one benchmark."""
        to_check = [
            "Certificate",
            "XD",
            "XI",
            "XU",
            "XG",
            "Neurons",
            "Activations",
            "Factors",
            "LLO",
            "Symmetric_Belt",
            "Time_Domain",
        ]
        for col in to_check:
            if df[col].unique().shape[0] > 1:
                warnings.warn("Column {} has multiple values".format(col))

    @staticmethod
    def get_success_ratio(df):
        """Get success ratio for given dataframe.

        Args:
            df (pd.DataFrame): dataframe to analyse

        Returns:
            float: success ratio
        """
        return df[df["Result"] == True].shape[0] / df.shape[0]

    @staticmethod
    def time_analysis(df):
        """Get time data for given dataframe.

        Args:
            df (pd.DataFrame): dataframe to analyse

        Returns:
            float: average time
        """
        mean = df["Total_Time"].mean()
        std = df["Total_Time"].std()
        Max = df["Total_Time"].max()
        Min = df["Total_Time"].min()

        time_stats = Stats(mean, std, Max, Min)

        return time_stats

    @staticmethod
    def loop_analysis(df):
        """Get loop data for given dataframe.

        Args:
            df (pd.DataFrame): dataframe to analyse

        Returns:
            float: average loops
        """
        mean = df["Loops"].mean()
        std = df["Loops"].std()
        Max = df["Loops"].max()
        Min = df["Loops"].min()

        loop_stats = Stats(mean, std, Max, Min)
        return loop_stats

    def table_main(self, benchmarks=[]):
        df = self.results
        if benchmarks == []:
            benchmarks = self.get_benchmarks()

        df = df[df["Benchmark_file"].isin(benchmarks)]

        df.rename(
            {"N_s": "$N_s$", "N_u": "$N_u$", "Benchmark_file": "Benchmark"},
            axis=1,
            inplace=True,
        )
        df["Benchmark"].replace({"_": "\_"}, inplace=True, regex=True)
        df["Activations"].replace({"_": "\_"}, inplace=True, regex=True)

        # grouped = df.groupby(["Benchmark_file"])
        vals = [
            "Total_Time",
            "Result",
        ]
        ind = [
            "Benchmark",
            "$N_s$",
            "$N_u$",
            "Certificate",
            "Neurons",
            "Activations",
        ]
        table = pd.pivot_table(
            df,
            values=vals,
            index=ind,
            aggfunc={"Total_Time": ["min", "mean", "max"], "Result": ratio},
        )
        table.rename(
            {
                "Total_Time": "$T$",
                "Width": "$N$",
                "mean": "$\mu$",
                "min": "$\min$",
                "max": "$\max$",
                "ratio": "$R$",
            },
            axis=1,
            inplace=True,
        )
        print(table)

        table.to_latex(
            "experiments/main_tab.tex",
            float_format="%.2f",
            bold_rows=False,
            escape=False,
            multicolumn_format="c",
        )


if __name__ == "__main__":
    a = Analyser()
    # benchmarks = a.get_benchmarks()
    # b = a.analyse_benchmark(benchmarks[0])
    # print(b)
    a.table_main()
