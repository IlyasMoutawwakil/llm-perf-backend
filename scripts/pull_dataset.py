import os

from huggingface_hub import snapshot_download, logging
from huggingface_hub.utils import disable_progress_bars


disable_progress_bars()
logging.set_verbosity_warning()


HF_TOKEN = os.environ.get("HF_TOKEN", None)


def main():
    snapshot_download(
        repo_id="optimum/llm-perf-dataset",
        repo_type="dataset",
        local_dir="dataset",
        token=HF_TOKEN,
    )


if __name__ == "__main__":
    main()
