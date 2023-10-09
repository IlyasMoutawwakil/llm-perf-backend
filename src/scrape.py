import glob
import json
import shutil
import numpy as np
import pandas as pd
from transformers import AutoConfig
from typing import List, Optional, Tuple
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from huggingface_hub import Repository, HfApi

METRICS = ["acc_norm", "acc_norm", "acc", "mc2"]
BENCHMARKS = ["arc:challenge", "hellaswag", "hendrycksTest", "truthfulqa:mc"]

OPEN_LLM_RESULTS_REPO = "https://huggingface.co/datasets/open-llm-leaderboard/results"
SCORES_DIR = "open-llm-leaderboard-results"


def parse_eval_result(json_filepath: str) -> Tuple[str, list[dict]]:
    with open(json_filepath) as fp:
        data = json.load(fp)

    for mmlu_k in [
        "harness|hendrycksTest-abstract_algebra|5",
        "hendrycksTest-abstract_algebra",
    ]:
        if mmlu_k in data["versions"] and data["versions"][mmlu_k] == 0:
            return None, []  # we skip models with the wrong version

    try:
        config = data.get("config", data.get("config_general", None))
        model = config.get("model_name", None)
        if model is None:
            model = config.get("model_args", None)

        model_sha = config.get("model_sha", "")
        eval_sha = config.get("lighteval_sha", "")
        model_split = model.split("/", 1)

        if len(model_split) == 1:
            org = None
            model = model_split[0]
            result_key = f"{model}_{model_sha}_{eval_sha}"
        else:
            org = model_split[0]
            model = model_split[1]
            model = f"{org}/{model}"
            result_key = f"{org}_{model}_{model_sha}_{eval_sha}"

        eval_results = []
        for benchmark, metric in zip(BENCHMARKS, METRICS):
            accs = np.array(
                [v[metric] for k, v in data["results"].items() if benchmark in k]
            )
            if accs.size == 0:
                continue
            mean_acc = round(np.mean(accs) * 100.0, 1)
            eval_results.append(
                {
                    "model": model,
                    "revision": "main",
                    "model_sha": model_sha,
                    "results": {benchmark: mean_acc},
                }
            )
        return result_key, eval_results

    except Exception as e:
        print(e, json_filepath)
        return None, []


def get_eval_results(eval_dir: str) -> List[dict]:
    json_filepaths = glob.glob(f"{eval_dir}/**/results*.json", recursive=True)

    eval_results = {}

    for json_filepath in json_filepaths:
        result_key, results = parse_eval_result(json_filepath)
        for eval_result in results:
            if result_key in eval_results.keys():
                eval_results[result_key]["results"].update(eval_result["results"])
            else:
                eval_results[result_key] = eval_result

    eval_results = [v for v in eval_results.values()]

    return eval_results


def get_model_metadata(model_name: str, revision: str) -> Optional[Tuple[str, int]]:
    try:
        auto_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=False,
            revision=revision,
        )
        with init_empty_weights():
            pretrained_model = AutoModelForCausalLM.from_config(config=auto_config)
        model_type = auto_config.model_type
        num_params = sum(
            p.numel() for p in pretrained_model.parameters() if p.requires_grad
        )

    except Exception as e:
        print(e)
        model_type = None
        num_params = None
    try:
        likes = HfApi().model_info(repo_id=model_name, revision=revision).likes
    except Exception as e:
        print(e)
        likes = None

    return model_type, num_params, likes


def scrape_open_llm_results():
    shutil.rmtree(SCORES_DIR, ignore_errors=True)
    Repository(
        local_dir=SCORES_DIR,
        clone_from=OPEN_LLM_RESULTS_REPO,
        repo_type="dataset",
    )
    open_llm_results_df = pd.DataFrame(get_eval_results(eval_dir=SCORES_DIR))
    open_llm_results_df["score"] = open_llm_results_df["results"].apply(
        lambda x: round(np.mean(list(x.values())), 1)
    )
    open_llm_results_df.drop(columns=["results"], inplace=True)
    open_llm_results_df.drop_duplicates(subset=["model"], inplace=True)
    open_llm_results_df["metadata"] = open_llm_results_df.apply(
        lambda row: get_model_metadata(row["model"], row["revision"]), axis=1
    )
    open_llm_results_df["model_type"] = open_llm_results_df["metadata"].apply(
        lambda x: x[0]
    )
    open_llm_results_df["num_params"] = open_llm_results_df["metadata"].apply(
        lambda x: x[1]
    )
    open_llm_results_df["num_likes"] = open_llm_results_df["metadata"].apply(
        lambda x: x[2]
    )
    open_llm_results_df.drop(columns=["metadata"], inplace=True)
    open_llm_results_df.to_csv("Open-LLM-Leaderboard.csv", index=False)
    print("Number of evaluated models:", len(open_llm_results_df))
    shutil.rmtree(SCORES_DIR, ignore_errors=True)


if __name__ == "__main__":
    scrape_open_llm_results()
