import os
import shutil
import pandas as pd

TEMPLATES_DIR = "llm-configs-templates"
CONFIGS_DIR = "llm-configs"


def populate_config_templates():
    df = pd.read_csv("Clustered-Open-LLM-Leaderboard.csv")
    shutil.rmtree(CONFIGS_DIR, ignore_errors=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    for template_file in os.listdir(TEMPLATES_DIR):
        template_text = open(f"{TEMPLATES_DIR}/{template_file}").read()
        template_name = template_file.split(".")[0]

        for _, row in df.iterrows():
            model = row["best_scored_model"]
            revision = "main"
            weight_class = row["weight_class"]
            model_type = row["model_type"]
            model_config = template_text.format(
                model=model,
                revision=revision,
                experiment_name=f"{template_name}_{weight_class}_{model_type}",
            )
            open(
                f"{CONFIGS_DIR}/{template_name}_{weight_class}_{model_type}.yaml", "w"
            ).write(model_config.format(**row))

    print(f"Number of created configs: {len(df) * len(os.listdir(TEMPLATES_DIR))}")


if __name__ == "__main__":
    populate_config_templates()
