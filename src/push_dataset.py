import os
from argparse import ArgumentParser

from huggingface_hub import HfApi


MACHINE = os.environ.get("MACHINE", "unknown")
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def push_dataset(dataset_id: str):
    HfApi().upload_folder(
        token=HF_TOKEN,
        repo_id=dataset_id,
        repo_type="dataset",
        folder_path="dataset",
        commit_message="Update dataset",
        delete_patterns=[f"{MACHINE}/*", f"{MACHINE}-failed/*", "open-llm.csv"],
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

    push_dataset(dataset_id)


if __name__ == "__main__":
    main()
