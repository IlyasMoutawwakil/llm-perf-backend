import os
import subprocess
from argparse import ArgumentParser

import pandas as pd

MACHINE = os.environ.get("MACHINE", "unknown")
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def get_models():
    open_llm = pd.read_csv("dataset/open-llm.csv")
    models = open_llm.sort_values("size")["model"].tolist()
    return models


def benchmark(config: str, model: str):
    print(f"Benchmarking model {model} with config {config} on machine {MACHINE}")
    out = subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "configs",
            "--config-name",
            config,
            f"model={model}",
            f"hydra.run.dir=dataset/{MACHINE}/{config}/{model}",
        ],
        capture_output=True,
    )

    if out.returncode != 0:
        print("Benchmarking failed")
        subprocess.run(
            [
                "mkdir",
                "-p",
                f"dataset/{MACHINE}-failed/{config}/{model}",
            ],
        )
        subprocess.run(
            [
                "cp",
                "-r",
                f"dataset/{MACHINE}/{config}/{model}/.",
                f"dataset/{MACHINE}-failed/{config}/{model}",
            ],
        )
    else:
        print("Benchmarking succeeded")
        subprocess.run(
            [
                "rm",
                "-rf",
                f"dataset/{MACHINE}-failed/{config}/{model}",
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

    args = parser.parse_args()
    config = args.config

    models = get_models()
    for model in models:
        benchmark(config, model)


if __name__ == "__main__":
    main()
