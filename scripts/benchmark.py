import os
import subprocess
from argparse import ArgumentParser

import pandas as pd
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub import login


HF_TOKEN = os.environ.get("HF_TOKEN", None)

if HF_TOKEN is not None:
    login(token=HF_TOKEN)


def get_models():
    open_llm_file = hf_hub_download(
        repo_type="dataset",
        filename="open-llm.csv",
        repo_id="optimum/llm-perf-dataset",
    )
    open_llm = pd.read_csv(open_llm_file)
    models = open_llm.sort_values("Size")["Model"].tolist()
    return models


def benchmark(config: str, model: str, debug: bool = False):
    print(f">Benchmarking model {model} with config {config} ...")
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

    models = get_models()
    config = args.config
    debug = args.debug

    for model in models:
        # run benchmark
        benchmark(config, model, debug)


if __name__ == "__main__":
    main()
