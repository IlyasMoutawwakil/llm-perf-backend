import os
import subprocess
from argparse import ArgumentParser

import pandas as pd
from huggingface_hub import login


HF_TOKEN = os.environ.get("HF_TOKEN", None)
HOSTNAME = os.environ.get("HOSTNAME", "UNKNOWN")

if HF_TOKEN is not None:
    login(token=HF_TOKEN)


def benchmark(config: str, model: str, debug: bool = False):
    if os.path.exists(f"dataset/{HOSTNAME}/{config}/{model}/inference_results.csv"):
        print(
            f">The benchmark of model {model} with config {config} already exists, skipping ..."
        )
        return

    print(f">Benchmarking model {model} with config {config} ...")
    try:
        out = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                "configs",
                "--config-name",
                config,
                f"model={model}",
            ],
            timeout=60 * 10,  # 10 minutes timeout
            capture_output=not debug,  # some privacy
        )
    except subprocess.TimeoutExpired:
        print(f">The benchmark of model {model} with config {config} timed out !")
        return

    # repeating myself because sometimes we get long outputs that overflow the terminal
    if out.returncode == 0:
        print(f">The benchmark of model {model} with config {config} succeeded !")
    else:
        print(f">The benchmark of model {model} with config {config} failed !")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode",
    )

    args = parser.parse_args()

    open_llm = pd.read_csv("dataset/open-llm.csv")
    custom_llm = pd.read_csv("dataset/custom-llm.csv")
    models = pd.concat([open_llm, custom_llm]).sort_values("Size")["Model"].unique()

    llm_perf = pd.read_csv(f"dataset/{HOSTNAME}/perf-report.csv")
    config = args.config
    debug = args.debug
    counter = 0

    for model in models:
        # check if model+config already benchmarked
        already_benchmarked = (
            (llm_perf["experiment_name"] == config) & (llm_perf["model"] == model)
        ).any()
        if already_benchmarked:
            print(
                f">The benchmark of model {model} with config {config} already exists, skipping ..."
            )
            continue

        counter += 1
        benchmark(config, model, debug)

        if counter == 10:
            # 10 new models per run ?
            break


if __name__ == "__main__":
    main()
