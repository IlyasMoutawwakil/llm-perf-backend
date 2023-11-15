from argparse import ArgumentParser
from pathlib import Path

from optimum_benchmark.aggregators.gather import gather


def aggregate_results():
    parser = ArgumentParser()

    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        help="Machine on which the benchmark is running",
    )

    args = parser.parse_args()

    machine = args.machine
    full_report = gather(root_folders=[Path(f"llm-perf-dataset/{machine}")])
    full_report.to_csv(f"llm-perf-dataset/{machine}/full-report.csv", index=False)


if __name__ == "__main__":
    aggregate_results()
