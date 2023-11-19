import os

from huggingface_hub import HfApi


HF_TOKEN = os.environ.get("HF_TOKEN", None)
MACHINE = os.environ.get("MACHINE", os.uname().nodename)


def main():
    HfApi().upload_folder(
        token=HF_TOKEN,
        repo_type="dataset",
        folder_path="dataset",
        commit_message="Update dataset",
        repo_id="optimum/llm-perf-dataset",
        delete_patterns=[
            f"{MACHINE}/*",
            f"{MACHINE}-failed/*",
        ],
    )


if __name__ == "__main__":
    main()
