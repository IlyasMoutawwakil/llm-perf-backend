import shutil
import os

MACHINE = "hf-dgx-01"
config = "pytorch+cuda+float64"
model = "01-ai/Yi-34B"

# create a failed directory
os.makedirs(f"dataset/{MACHINE}/{config}/{model}", exist_ok=True)

shutil.move(
    f"dataset/{MACHINE}/{config}/{model}",
    f"dataset/{MACHINE}-failed/{config}/{model}",
)
