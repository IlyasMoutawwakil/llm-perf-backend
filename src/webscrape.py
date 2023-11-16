# adapted from https://github.com/Weyaxi/scrape-open-llm-leaderboard

import requests
import json
import os

from transformers import AutoConfig
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://huggingfaceh4-open-llm-leaderboard.hf.space/"
TOKEN = os.environ.get("HF_TOKEN", None)


def get_json_data():
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, "html.parser")
    script_elements = soup.find_all("script")
    json_format_data = json.loads(str(script_elements[1])[31:-10])
    return json_format_data


def get_list_data(data):
    for component_index in range(10, 50, 1):
        try:
            result_list = []
            i = 0
            while True:
                try:
                    results = data["components"][component_index]["props"]["value"][
                        "data"
                    ][i]
                    type_of_emoji = data["components"][component_index]["props"][
                        "value"
                    ]["data"][i][0]
                    try:
                        results_json = {
                            "T": type_of_emoji,
                            "Model": results[-1],
                            "Average ⬆️": results[2],
                            "ARC": results[3],
                            "HellaSwag": results[4],
                            "MMLU": results[5],
                            "TruthfulQA": results[6],
                            "Winogrande": results[7],
                            "GSM8K": results[8],
                            "DROP": results[9],
                            "Type": results[10],
                            "Precision": results[11],
                            "Hub License": results[12],
                            "#Params (B)": results[13],
                            "Hub ❤️": results[14],
                            "Model Sha": results[16],
                        }
                    except IndexError:
                        break
                    result_list.append(results_json)
                    i += 1
                except IndexError:
                    return result_list
        except (KeyError, TypeError):
            continue

    return result_list


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


def process_leaderboard(df):
    df.drop(columns=["T"], inplace=True)
    df.rename(
        columns={
            "Type": "type",
            "Model": "model",
            "Precision": "precision",
            "Average ⬆️": "avg_score",
            "Hub ❤️": "likes",
            "#Params (B)": "size(B)",
            "Hub License": "licence",
        },
        inplace=True,
    )
    df.dropna(subset=["model", "avg_score"], inplace=True)
    df = df[df["type"] == "pretrained"]
    df = df[
        df["precision"].isin(
            ["torch.float32", "torch.float16", "torch.bfloat16", "8bit", "4bit"]
        )
    ]
    df.sort_values(by="precision", ascending=False, inplace=True)
    df.drop_duplicates(subset=["model"], keep="first", inplace=True)
    df["architecture"] = df["model"].apply(get_model_arch)
    df.sort_values(by=["avg_score"], ascending=False, inplace=True)
    df = df[["model", "size(B)", "architecture", "avg_score", "likes"]]
    return df


def main():
    json_data = get_json_data()
    list_data = get_list_data(json_data)
    open_llm_df = pd.DataFrame(list_data)
    open_llm_df = process_leaderboard(open_llm_df)
    os.makedirs("dataset", exist_ok=True)
    open_llm_df.to_csv("dataset/open-llm.csv", index=False)


if __name__ == "__main__":
    main()
