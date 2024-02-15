import os
import datetime
import subprocess

import pandas as pd
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", None)

if HF_TOKEN is not None:
    login(token=HF_TOKEN)

HOSTNAME = os.environ.get("HOSTNAME", "UNKNOWN")


def main():
    run_id = datetime.datetime.now().isoformat()
    print(">Gathering local performance report")
    out = subprocess.run(
        [
            "optimum-report",
            "gather",
            "--root-folders",
            f"dataset/{HOSTNAME}",
            "--save-file",
            f"dataset/{HOSTNAME}/perf-report-{run_id}.csv",
        ],
        capture_output=True,
    )

    if out.returncode == 0:
        print(">Report generation succeeded")
    else:
        raise Exception("Report generation failed")

    print(">Updating performance report")
    old_entries_df = pd.read_csv(f"dataset/{HOSTNAME}/perf-report.csv")
    new_entries_df = pd.read_csv(f"dataset/{HOSTNAME}/perf-report-{run_id}.csv")
    perf_report = pd.concat([old_entries_df, new_entries_df], ignore_index=True)
    perf_report.to_csv(f"dataset/{HOSTNAME}/perf-report.csv", index=False)

    print(">Uploading llm-perf-dataset")
    HfApi().upload_folder(
        repo_id="optimum/llm-perf-dataset",
        allow_patterns=[f"{HOSTNAME}/*"],
        commit_message="Update dataset",
        folder_path="dataset",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
