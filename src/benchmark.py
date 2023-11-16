import subprocess
from argparse import ArgumentParser

import pandas as pd


def benchmark(config: str, model: str, machine: str):
    print(f"Benchmarking model {model} on machine {machine} with config {config}")
    subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "configs",
            "--config-name",
            config,
            f"model={model}",
            f"hydra.run.dir=dataset/{machine}/{config}/{model}",
        ],
    )


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config directory",
    )
    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        help="Machine on which the benchmark is running",
    )

    args = parser.parse_args()

    config = args.config
    machine = args.machine

    MODELS = pd.read_csv("dataset/open-llm.csv")["Model"].tolist()
    for model in MODELS:
        benchmark(config, model, machine)


if __name__ == "__main__":
    main()
