import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def pull_dataset():
    argparser = ArgumentParser()

    argparser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="optimum/llm-perf-dataset",
    )
    argparser.add_argument(
        "--folder",
        type=str,
        help="Folder name",
        default="llm-perf-dataset",
    )

    args = argparser.parse_args()

    repo_id = args.dataset
    local_dir = args.folder

    snapshot_download(
        token=HF_TOKEN,
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
    )


if __name__ == "__main__":
    pull_dataset()
