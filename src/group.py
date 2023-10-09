import pandas as pd


def cluster_by_weight_class_and_model_type():
    df = pd.read_csv("Open-LLM-Leaderboard.csv")
    df = df[df.notnull().all(axis=1)]
    df["num_params"] = df["num_params"].apply(
        lambda x: f"~{int(x/1e9)}B" if x > 1e9 else "<1B"
    )
    grouped_llms = df.groupby(by=["model_type", "weight_class"], as_index=False).agg(
        models=("model", list),
        scores=("score", list),
        likes=("num_likes", list),
        params=("num_params", list),
    )
    grouped_llms.rename(columns={"params": "num_params"}, inplace=True)
    grouped_llms["best_score"] = grouped_llms["scores"].apply(max)
    grouped_llms["best_scored_model"] = grouped_llms[
        ["models", "scores", "best_score"]
    ].apply(lambda x: x.iloc[0][x.iloc[1].index(x.iloc[2])], axis=1)
    grouped_llms["most_likes"] = grouped_llms["likes"].apply(max)
    grouped_llms["most_liked_model"] = grouped_llms[
        ["models", "likes", "most_likes"]
    ].apply(lambda x: x.iloc[0][x.iloc[1].index(x.iloc[2])], axis=1)
    grouped_llms["biggest_#params"] = grouped_llms["#params"].apply(max)
    grouped_llms["biggest_#params_model"] = grouped_llms[
        ["models", "#params", "biggest_#params"]
    ].apply(lambda x: x.iloc[0][x.iloc[1].index(x.iloc[2])], axis=1)
    grouped_llms["models_count"] = grouped_llms["models"].apply(len)
    grouped_llms.to_csv("Clustered-Open-LLM-Leaderboard.csv", index=False)
    return grouped_llms


if __name__ == "__main__":
    cluster_by_weight_class_and_model_type()
