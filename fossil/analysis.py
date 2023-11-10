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


from fossil.consts import CegisConfig, CertificateType, ACTIVATION_NAMES, PROPERTIES
from fossil import certificate

"""Post processing of results module."""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Change this line for different works
JOURNAL_RESULTS_FILE = "/results/results.csv"
TOOL_RESULTS_FILE = "/results/raw_tool_results.csv"
DRF = DEFAULT_RESULT_FILE = BASE_DIR + TOOL_RESULTS_FILE


@dataclasses.dataclass
class AnalysisConfig:
    """Class for storing configuration for analysis."""

    results_file: str = DRF
    benchmarks: tuple[str] = ()
    output_type = ["tex", "csv"]  # "csv" or "tex" or "md"
    output_file: str = "results/main_tab"


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
    "latex",
    "domains",
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
    return int(100 * x.sum() / x.count())


def _get_model_name(model):
    """Returns the class name of the model, or the open_loop class name if model is a GeneralClosedLoopModel."""
    try:
        name = model.__name__
    except AttributeError:
        name = type(model(None).open_loop).__name__
    return name


def get_activations(acts):
    return "[" + ",".join([ACTIVATION_NAMES[act] for act in acts]) + "]"


def get_property_from_certificate(certificate):
    return PROPERTIES[certificate]


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

        if (
            config.CERTIFICATE == CertificateType.STABLESAFE
            or config.CERTIFICATE == CertificateType.RAR
        ):
            alt_layers = config.N_HIDDEN_NEURONS_ALT
            alt_act = get_activations(config.ACTIVATION_ALT)
        else:
            alt_layers = None
            alt_act = None

        latex = benchmark_to_latex(config)
        domains = domains_to_string(config)

        result = [
            get_property_from_certificate(config.CERTIFICATE),
            config.N_VARS,
            N_u,
            result.res,
            model_name,
            cegis_stats.seed,
            latex,
            domains,
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
        print(config.results_file)
        self.results = pd.read_csv(config.results_file)
        self.output_type = config.output_type
        self.output_file = config.output_file  # + "." + self.output_type

    def get_benchmarks(self):
        """Get list of benchmarks in results file."""
        return self.results["Benchmark_file"].unique()

    def table_main(self, benchmarks=[]):
        df = self.results

        df.rename(
            {"N_s": "$N_s$", "N_u": "$N_u$", "Benchmark_file": "Model"},
            axis=1,
            inplace=True,
        )
        df["Model"].replace({"_": "\_"}, inplace=True, regex=True)
        df["Activations"].replace({"_": "\_"}, inplace=True, regex=True)
        df["Alt_Activations"].replace({"_": "\_"}, inplace=True, regex=True)

        df["Activations"] = df["Activations"] + ", " + df["Alt_Activations"].fillna("")
        df["Neurons"] = df["Neurons"] + ", " + df["Alt_Neurons"].fillna("")
        df["Certificate"] = df["Certificate"].str.replace("RWS", "RWA")
        df["Certificate"] = df["Certificate"].str.replace("RSWS", "RSWA")

        # grouped = df.groupby(["Benchmark_file"])
        vals = [
            "Total_Time",
            "Learner_Time",
            "Result",
        ]
        ind = [
            "latex",
            "domains",
            "Model",
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
            aggfunc={
                "Total_Time": ["min", "mean", "max"],
                "Learner_Time": ["min", "mean", "max"],
                "Result": ratio,
            },
            sort=False,
        )

        table.rename(
            {
                "Total_Time": "$T$ (s)",
                "Learner_Time": "$T_L$ (s)",
                "Neurons": "$W$",
                "mean": "$\mu$",
                "min": "$\min$",
                "max": "$\max$",
                "ratio": "$S$",
                "Certificate": "Property",
                "Result": "Successful (\%)",
            },
            axis=1,
            inplace=True,
        )
        table.reset_index(inplace=True)
        table.index = table.index + 1
        # df_to_latex(
        #     table, self.output_file
        # )  # Creates a latex file with all the benchmarks
        table.drop(["latex", "Model", "domains"], axis=1, inplace=True)
        table["$T$ (s)"] = (
            table["$T$ (s)"].round(2).astype(str)
            + " ("
            + table["$T_L$ (s)"].round(2).astype(str)
            + ")"
        )
        table.drop(["$T_L$ (s)"], axis=1, inplace=True)
        pd.options.display.float_format = "{:,.2f}".format
        print(table)

        if "tex" in self.output_type:
            table.to_latex(
                self.output_file + ".tex",
                float_format="%.2f",
                bold_rows=False,
                escape=False,
                multicolumn_format="c",
            )
        if "csv" in self.output_type:
            table.columns = table.columns.map(" ".join)
            table.to_csv(self.output_file + ".csv")
        if "md" in self.output_type:
            table.to_markdown(self.output_file + ".md")


def benchmark_to_latex(config: CegisConfig):
    """Convert a benchmark to latex format.

    Args:
        config (CegisConfig): cegis config

    Returns:
        str: latex string
    """
    s = ""
    model = config.SYSTEM
    try:
        s += model().to_latex() + " \\n "
    except TypeError:
        s += model(None).open_loop.to_latex() + " \\n "
    domains = config.DOMAINS
    cert = config.CERTIFICATE
    if cert == CertificateType.LYAPUNOV:
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
    elif cert == CertificateType.ROA:
        s += "XI: {} ".format(domains[certificate.XI].to_latex()) + "\\n"
    elif cert in (CertificateType.BARRIER, CertificateType.BARRIERALT):
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
        s += "XI: {} ".format(domains[certificate.XI].to_latex()) + "\\n"
        s += "XU: {} ".format(domains[certificate.XU].to_latex()) + "\\n"
    elif cert == CertificateType.RWS:
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
        s += "XS: {} ".format(domains[certificate.XS].to_latex()) + "\\n"
        s += "XI: {} ".format(domains[certificate.XI].to_latex()) + "\\n"
        s += "XG: {} ".format(domains[certificate.XG].to_latex()) + "\\n"
    elif cert == CertificateType.RSWS:
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
        s += "XS: {} ".format(domains[certificate.XS].to_latex()) + "\\n"
        s += "XI: {} ".format(domains[certificate.XI].to_latex()) + "\\n"
        s += "XG: {} ".format(domains[certificate.XG].to_latex()) + "\\n"
    elif cert == CertificateType.STABLESAFE:
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
        s += "XI: {} ".format(domains[certificate.XI].to_latex()) + "\\n"
        s += "XU: {} ".format(domains[certificate.XU].to_latex()) + "\\n"
    elif cert == CertificateType.RAR:
        s += "XD: {} ".format(domains[certificate.XD].to_latex()) + "\\n"
        s += "XS: {} ".format(domains[certificate.XS].to_latex()) + "\\n"
        s += "XI: {} ".format(domains[certificate.XI].to_latex()) + "\\n"
        s += "XG: {} ".format(domains[certificate.XG].to_latex()) + "\\n"
        s += "XF: {} ".format(domains[certificate.XF].to_latex()) + "\\n"

    # Add activations, alt activations, control
    s += "\\nActivation: {}, Neurons: {}".format(
        get_activations(config.ACTIVATION), config.N_HIDDEN_NEURONS
    )
    if cert in (CertificateType.STABLESAFE, CertificateType.RAR):
        s += "\\n"
        s += "AltActivation: {}, AltNeurons: {}".format(
            get_activations(config.ACTIVATION_ALT), config.N_HIDDEN_NEURONS_ALT
        )
    if config.CTRLAYER is not None:
        s += "\\n"
        s += "CTRLActivation: {}, Neurons: {}".format(
            get_activations(config.CTRLACTIVATION), config.CTRLAYER
        )
    return s


def domains_to_string(config: CegisConfig):
    """Convert a list of domains to a string.

    Args:
        domains (dict[str, Any]): domains

    Returns:
        str: string representation of domains
    """
    s = ""
    model = config.SYSTEM
    try:
        s += "$" + model().to_latex() + "$" + " \\n \\n "
    except TypeError:
        s += model(None).open_loop.to_latex() + " \\n \\n "
    domains = config.DOMAINS
    cert = config.CERTIFICATE
    for lab, dom in domains.items():
        if lab not in certificate.BORDERS:
            s += "${}: {} $\\n".format(lab, dom.__repr__()) + "\\n"
            s = s.replace("lie", "XD")
            s = s.replace("init", "XI")
            s = s.replace("unsafe", "XU")
            s = s.replace("safe", "XS")
            s = s.replace("goal", "XG")
            s = s.replace("final", "XF")

        # Add activations, alt activations, control
    s += "\\nActivation: {}, Neurons: {}".format(
        get_activations(config.ACTIVATION), config.N_HIDDEN_NEURONS
    )
    if cert in (CertificateType.STABLESAFE, CertificateType.RAR):
        s += "\\n"
        s += "AltActivation: {}, AltNeurons: {}".format(
            get_activations(config.ACTIVATION_ALT), config.N_HIDDEN_NEURONS_ALT
        )
    if config.CTRLAYER is not None:
        s += "\\n"
        s += "CTRLActivation: {}, Neurons: {}".format(
            get_activations(config.CTRLACTIVATION), config.CTRLAYER
        )
    return s


def df_to_latex(results_frame: pd.DataFrame, output_file: str):
    """Gets all latex representations of benchmarks in dataframe and export to latex file.

    Args:
        results_file (str): results file
    """
    df = results_frame
    s = ""
    # add standard latex header
    # s += "\\documentclass{standalone}\n"
    # s += "\\usepackage{amsmath}\n"
    # s += "\\usepackage{amssymb}\n"
    # s += "\\begin{document}\n"

    for index, row in df.iterrows():
        benchmark = row["domains"].iloc[0]
        cert = row["Certificate"].iloc[0]
        model = row["Model"].iloc[0]

        benchmark = benchmark.replace("\\n", "\n")
        s += "\\textbf{{ [{}] {} {} }}\n\n".format(index, model, cert)
        s += benchmark + "\n\n"

    # s += "\\end{document}\n"
    latex_file = output_file + "_benchmarks.tex"
    with open(latex_file, "w") as f:
        f.write(s)


if __name__ == "__main__":
    c = AnalysisConfig(output_file="results/tool_paper_tab")
    a = Analyser(config=c)
    a.table_main()
