import os
from argparse import ArgumentParser

from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def push_dataset(dataset_id: str, dataset_path: str):
    HfApi().upload_folder(
        token=HF_TOKEN,
        repo_id=dataset_id,
        delete_patterns="*",
        repo_type="dataset",
        folder_path=dataset_path,
        commit_message="Update dataset",
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

    push_dataset(dataset_id, dataset_path)


if __name__ == "__main__":
    main()
