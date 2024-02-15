import os
from huggingface_hub import snapshot_download, logging
from huggingface_hub.utils import disable_progress_bars


disable_progress_bars()
logging.set_verbosity_warning()

HOSTNAME = os.environ.get("HOSTNAME", "UNKNOWN")


def main():
    snapshot_download(
        repo_id="optimum/llm-perf-dataset",
        allow_patterns=[
            "open-llm.csv",
            "custom-llm.csv",
            f"{HOSTNAME}/perf-report.csv",
        ],
        local_dir="dataset",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
