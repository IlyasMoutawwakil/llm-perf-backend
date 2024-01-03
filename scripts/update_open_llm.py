import os

from huggingface_hub import HfApi, login
from transformers import AutoConfig
import pandas as pd

HF_TOKEN = os.environ.get("HF_TOKEN", None)

if HF_TOKEN is not None:
    login(token=HF_TOKEN)

ALL_COLUMNS = {
    "Type": "Type",
    "Model": "Model",
    "Precision": "Precision",
    "Average ⬆️": "Score",
    "Hub ❤️": "Likes",
    "#Params (B)": "Size",
    "Hub License": "Licence",
}


def get_model_arch(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        return config.model_type
    except Exception:
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            return config.model_type
        except Exception:
            print(f"Failed to get {model_name} even with trust_remote_code=True")
            return None


def process_open_llm():
    df = pd.read_csv("open-llm-leaderboard.csv")

    # Rename columns
    df.rename(columns=ALL_COLUMNS, inplace=True)
    df = df[ALL_COLUMNS.values()]

    df.dropna(subset=["Model", "Score"], inplace=True)
    print(f"Found {len(df)} models in the Open-LLM leaderboard")

    # Filter out fine-tuned variants
    df = df[df["Type"] == "pretrained"]
    print(f"Found {len(df)} models after filtering pretrained variants")

    # Filter out quantized variants (we do that ourselves)
    df = df[~df["Precision"].isin(["GPTQ"])]
    df.sort_values(by="Precision", ascending=False, inplace=True)
    df.drop_duplicates(subset=["Model"], keep="first", inplace=True)
    print(f"Found {len(df)} models after filtering quantized variants")

    # Get model architecture
    df["Arch"] = df["Model"].apply(get_model_arch)

    # sort by score
    df.sort_values(by=["Score"], ascending=False, inplace=True)

    return df


def main():
    processed_open_llm_df = process_open_llm()
    processed_open_llm_df.to_csv("open-llm.csv", index=False)

    if len(processed_open_llm_df) > 0:
        HfApi().upload_file(
            commit_message="Update open models list",
            repo_id="optimum/llm-perf-dataset",
            path_or_fileobj="open-llm.csv",
            path_in_repo="open-llm.csv",
            repo_type="dataset",
        )
    else:
        raise ValueError("No models found")


if __name__ == "__main__":
    main()
