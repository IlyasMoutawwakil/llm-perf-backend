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

    print(">Uploading dataset to Hub")
    HfApi().upload_folder(
        folder_path=f"dataset",
        commit_message="Update dataset",
        allow_patterns=[f"{HOSTNAME}/*"],
        repo_id="optimum/llm-perf-dataset",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
