from datasets import Dataset, Image
import pandas as pd
import os
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HF_HUB_WRITE_TOKEN"))

image_folder = r"C:\Users\hp\Desktop\sem8\genAI\proj\finetuning-set-3"
csv_path = os.path.join(image_folder, "metadata.csv")

df = pd.read_csv(csv_path)
df["image"] = df["file_name"].apply(lambda fn: os.path.join(image_folder, fn))
df = df[["image", "caption"]]

dataset = Dataset.from_pandas(df).cast_column("image", Image())

dataset.push_to_hub("ibrahim7004/ghibli-images-for-SD1.5", split="train")
