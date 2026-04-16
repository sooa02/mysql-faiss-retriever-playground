from src import config
from huggingface_hub import login
from datasets import load_dataset

hf_token = config.get_env('HF_TOKEN')
login(token = hf_token)


def get_dataset(data_split = "train"):
    data = load_dataset("Youseff1987/resume-matching-dataset-v2", split=data_split)
    return data