import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from preprocess import preprocess_text, EmotionDataset


print("Loading trained model and tokenizer...")
model_path = "./results/final_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


print("Loading and preprocessing validation data...")
dev_df = pd.read_csv('public_data/dev/track_a/eng_a.csv')
emotions = ['Joy', 'Sadness', 'Surprise', 'Fear', 'Anger']

dev_df['text'] = dev_df['text'].apply(preprocess_text)


print("Preparing validation dataset...")
dev_dataset = EmotionDataset(dev_df['text'].tolist(), np.zeros((len(dev_df), len(emotions))), tokenizer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


print("Generating predictions...")
predictions = []

for batch in DataLoader(dev_dataset, batch_size=8):
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.sigmoid(logits).cpu().numpy()
    predictions.extend(preds)


predictions = (np.array(predictions) > 0.5).astype(int)


print("Saving predictions...")
for i, emotion in enumerate(emotions):
    dev_df[emotion] = predictions[:, i]

output_file = "./results/dev_predictions.csv"
dev_df.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'.")
