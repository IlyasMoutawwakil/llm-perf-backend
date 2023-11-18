import os
import shutil
import subprocess
from argparse import ArgumentParser

from huggingface_hub.file_download import hf_hub_download
import pandas as pd

MACHINE = os.environ.get("MACHINE", "unknown")
HF_TOKEN = os.environ.get("HF_TOKEN", None)


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
    if os.path.exists(f"dataset/{MACHINE}/{config}/{model}/inference_results.csv"):
        print(f">Skipping model {model} with config {config} on machine {MACHINE}")
        return

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
        capture_output=True,
    )

    if out.returncode != 0:
        print(">Benchmarking failed")
        shutil.move(
            f"dataset/{MACHINE}/{config}/{model}",
            f"dataset/{MACHINE}-failed/{config}/{model}",
        )
    else:
        print(">Benchmarking succeeded")
        if os.path.exists(f"dataset/{MACHINE}-failed/{config}/{model}"):
            shutil.rmtree(f"dataset/{MACHINE}-failed/{config}/{model}")


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
