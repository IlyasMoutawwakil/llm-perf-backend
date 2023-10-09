from flatten_dict import flatten
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd


def gather_inference_report(root_folder: Path) -> pd.DataFrame:
    # key is path to inference file as string, value is dataframe
    inference_dfs = {
        f.parent.absolute().as_posix(): pd.read_csv(f)
        for f in root_folder.glob("**/inference_results.csv")
    }
    # key is path to config file as string, value is flattened dict
    config_dfs = {
        f.parent.absolute()
        .as_posix(): pd.DataFrame.from_dict(
            flatten(OmegaConf.load(f), reducer="dot"), orient="index"
        )
        .T
        for f in root_folder.glob("**/hydra_config.yaml")
        if f.parent.absolute().as_posix() in inference_dfs.keys()
    }
    if len(inference_dfs) == 0:
        inference_dfs = {
            f.parent.absolute().as_posix(): pd.DataFrame(index=[0])
            for f in root_folder.glob("**/hydra_config.yaml")
        }
        config_dfs = {
            f.parent.absolute()
            .as_posix(): pd.DataFrame.from_dict(
                flatten(OmegaConf.load(f), reducer="dot"), orient="index"
            )
            .T
            for f in root_folder.glob("**/hydra_config.yaml")
            if f.parent.absolute().as_posix() in inference_dfs.keys()
        }
    # Merge inference and config dataframes
    inference_reports = [
        config_dfs[name].merge(inference_dfs[name], left_index=True, right_index=True)
        for name in inference_dfs.keys()
    ]
    # Concatenate all reports
    inference_report = pd.concat(inference_reports, axis=0, ignore_index=True)
    return inference_report


for benchmark in Path("llm-experiments").glob("*"):
    try:
        benchmark_experiments = gather_inference_report(benchmark)
        benchmark_experiments.to_csv(f"reports/{benchmark.name}.csv")
        # save markdown table
        with open(f"reports/{benchmark.name}.md", "w") as f:
            f.write(benchmark_experiments.to_markdown())
    except Exception as e:
        print(f"Error while processing {benchmark.name}")
        print(e)
