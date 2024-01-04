import os
import subprocess

from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", None)

if HF_TOKEN is not None:
    login(token=HF_TOKEN)

HOSTNAME = os.environ.get("HOSTNAME", "UNKNOWN")


def main():
    out = subprocess.run(
        [
            "optimum-report",
            "gather",
            "--root-folders",
            f"dataset/{HOSTNAME}",
            "--save-file",
            f"dataset/{HOSTNAME}/perf-report.csv",
        ],
        capture_output=True,
    )

    if out.returncode == 0:
        print(">Report generation succeeded")
    else:
        raise Exception("Report generation failed")

    HfApi().upload_folder(
        repo_type="dataset",
        folder_path="dataset",
        commit_message="Update dataset",
        repo_id="optimum/llm-perf-dataset",
        delete_patterns=[f"{HOSTNAME}/*"],
    )


if __name__ == "__main__":
    main()
