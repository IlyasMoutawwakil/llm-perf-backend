from argparse import ArgumentParser
import subprocess

import pandas as pd


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

    models = pd.read_csv("llm-perf-dataset/open-llm.csv")["Model"].tolist()

    for model in models:
        print(f"Benchmarking model {model}")
        subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                "configs",
                "--config-name",
                config,
                f"model={model}",
                f"hydra.run.dir=llm-perf-dataset/{machine}/{config}/{model}",
            ],
        )


if __name__ == "__main__":
    main()
