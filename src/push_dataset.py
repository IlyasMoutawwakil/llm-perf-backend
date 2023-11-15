import os
from argparse import ArgumentParser

from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def push_dataset():
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
    folder_path = args.folder

    HfApi().upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        commit_message="Update dataset",
        repo_type="dataset",
        delete_patterns="*",
        token=HF_TOKEN,
    )


if __name__ == "__main__":
    push_dataset()
