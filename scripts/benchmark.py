import os
import shutil
import subprocess
from argparse import ArgumentParser

import pandas as pd
from huggingface_hub.file_download import hf_hub_download


HF_TOKEN = os.environ.get("HF_TOKEN", None)


def get_models():
    open_llm_file = hf_hub_download(
        token=HF_TOKEN,
        repo_type="dataset",
        filename="open-llm.csv",
        repo_id="optimum/llm-perf-dataset",
    )
    open_llm = pd.read_csv(open_llm_file)
    models = open_llm.sort_values("Size")["Model"].tolist()
    return models


def get_available_gpus():
    available_gpus = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    available_gpus = [int(gpu_id) for gpu_id in available_gpus if gpu_id != ""]

    return available_gpus


def get_pid_list(gpu_id: int):
    pid_list = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    pid_list = [int(pid) for pid in pid_list if pid != ""]

    return pid_list


def get_gpu_name(gpu_id: int):
    gpu_name = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=gpu_name",
                "--format=csv,noheader",
            ]
        )
        .decode("utf-8")
        .strip()
    )

    return gpu_name


def get_idle_gpu():
    available_gpus = get_available_gpus()

    for gpu_id in available_gpus:
        pid_list = get_pid_list(gpu_id)
        gpu_name = get_gpu_name(gpu_id)

        if len(pid_list) == 0 and gpu_name != "NVIDIA DGX Display":
            return gpu_id

    return None


def benchmark(config: str, model: str, machine: str, debug: bool = False):
    # skip if inference_results.csv already exists
    if os.path.exists(f"dataset/{machine}/{config}/{model}/inference_results.csv"):
        print(f">Model {model} with config {config} already benchmarked")
        return

    # remove failed experiment (must be failed since inference_results.csv is missing)
    if os.path.exists(f"dataset/{machine}/{config}/{model}"):
        shutil.rmtree(f"dataset/{machine}/{config}/{model}")

    # get idle gpu
    idle_gpu_id = get_idle_gpu()
    if idle_gpu_id is None:
        print(">No idle GPU found")
        return

    print(f">Benchmarking model {model} with config {config} on GPU {idle_gpu_id}")
    out = subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "configs",
            "--config-name",
            config,
            f"model={model}",
            f"hub_kwargs.token={HF_TOKEN}",
            f"hydra.run.dir=dataset/{machine}/{config}/{model}",
            f"hydra.job.env_set.CUDA_VISIBLE_DEVICES={idle_gpu_id}",
        ],
        capture_output=not debug,
    )

    if out.returncode == 0:
        print(">Benchmarking succeeded")
        # remove previous failed experiment if it exists
        if os.path.exists(f"dataset/{machine}-failed/{config}/{model}"):
            shutil.rmtree(f"dataset/{machine}-failed/{config}/{model}")
    else:
        print(">Benchmarking failed")
        # remove previous failed experiment to put the new one in its place
        if os.path.exists(f"dataset/{machine}-failed/{config}/{model}"):
            shutil.rmtree(f"dataset/{machine}-failed/{config}/{model}")
        # move the new failed experiment to the failed folder
        shutil.move(
            f"dataset/{machine}/{config}/{model}",
            f"dataset/{machine}-failed/{config}/{model}",
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
        "--debug",
        action="store_true",
        help="Debug mode",
    )

    args = parser.parse_args()

    debug = args.debug
    config = args.config
    models = get_models()
    machine = os.uname().nodename

    for model in models:
        benchmark(config, model, machine, debug=debug)


if __name__ == "__main__":
    main()
