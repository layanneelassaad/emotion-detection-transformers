import torch
from torch.utils.data import Dataset
import re


def preprocess_text(text):
    text = text.lower() #lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  #remove special chars
    text = re.sub(r"\s+", " ", text).strip()    #remove extra space
    return text


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128): #tokenize and save labels
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
