import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download, logging
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
logging.set_verbosity_warning()

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def pull_dataset(dataset_id: str):
    snapshot_download(
        token=HF_TOKEN,
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir="dataset",
    )


def main():
    argparser = ArgumentParser()

    argparser.add_argument(
        "--dataset-id",
        type=str,
        help="Dataset name",
        default="optimum/llm-perf-dataset",
    )

    args = argparser.parse_args()
    dataset_id = args.dataset_id

    pull_dataset(dataset_id)


if __name__ == "__main__":
    main()
