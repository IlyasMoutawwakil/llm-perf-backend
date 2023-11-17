import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download, logging
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
logging.set_verbosity_warning()

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def pull_dataset(dataset_id: str, dataset_path: str):
    snapshot_download(
        token=HF_TOKEN,
        repo_id=dataset_id,
        local_dir=dataset_path,
        repo_type="dataset",
    )


def main():
    argparser = ArgumentParser()

    argparser.add_argument(
        "--dataset-id",
        type=str,
        help="Dataset name",
        default="optimum/llm-perf-dataset",
    )
    argparser.add_argument(
        "--dataset-path",
        type=str,
        help="Dataset path",
        default="dataset",
    )

    args = argparser.parse_args()
    dataset_id = args.dataset_id
    dataset_path = args.dataset_path

    pull_dataset(dataset_id, dataset_path)


if __name__ == "__main__":
    main()
