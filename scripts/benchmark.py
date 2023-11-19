import os
import shutil
import subprocess
from argparse import ArgumentParser

import pandas as pd
from huggingface_hub.file_download import hf_hub_download


HF_TOKEN = os.environ.get("HF_TOKEN", None)
MACHINE = os.environ.get("MACHINE", os.uname().nodename)


def get_models():
    open_llm_file = hf_hub_download(
        repo_type="dataset",
        filename="open-llm.csv",
        use_auth_token=HF_TOKEN,
        repo_id="optimum/llm-perf-dataset",
    )
    open_llm = pd.read_csv(open_llm_file)
    models = open_llm.sort_values("Size")["Model"].tolist()
    return models


def benchmark(config: str, model: str):
    # skip if inference_results.csv already exists
    if os.path.exists(f"dataset/{MACHINE}/{config}/{model}/inference_results.csv"):
        print(f">Skipping model {model} with config {config} on machine {MACHINE}")
        return

    # remove failed experiment (must be failed since inference_results.csv is missing)
    if os.path.exists(f"dataset/{MACHINE}/{config}/{model}"):
        shutil.rmtree(f"dataset/{MACHINE}/{config}/{model}")

    print(f">Benchmarking model {model} with config {config} on machine {MACHINE}")
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
    )

    if out.returncode == 0:
        print(">Benchmarking succeeded")

        # remove previous failed experiment if it exists
        if os.path.exists(f"dataset/{MACHINE}-failed/{config}/{model}"):
            shutil.rmtree(f"dataset/{MACHINE}-failed/{config}/{model}")
    else:
        print(">Benchmarking failed")

        # remove previous failed experiment to put the new one in its place
        if os.path.exists(f"dataset/{MACHINE}-failed/{config}/{model}"):
            shutil.rmtree(f"dataset/{MACHINE}-failed/{config}/{model}")

        # move the new failed experiment to the failed folder
        shutil.move(
            f"dataset/{MACHINE}/{config}/{model}",
            f"dataset/{MACHINE}-failed/{config}/{model}",
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
