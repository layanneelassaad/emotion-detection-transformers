import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from preprocess import preprocess_text, EmotionDataset


print("Loading and preprocessing training and validation data...")
train_df = pd.read_csv('public_data/train/track_a/eng.csv')
dev_df = pd.read_csv('public_data/dev/track_a/eng_a.csv')


train_df['text'] = train_df['text'].apply(preprocess_text)
dev_df['text'] = dev_df['text'].apply(preprocess_text)


emotions = ['Joy', 'Sadness', 'Surprise', 'Fear', 'Anger']
label_counts = train_df[emotions].sum()

print("Counts of each label in the training set:")
print(label_counts)
train_labels = train_df[emotions].values
dev_labels = dev_df[emotions].values

print("Computing class weights...")
label_weights = []
for i in range(train_labels.shape[1]):  #weights per label
    positive_weight = (train_labels.shape[0] - train_labels[:, i].sum()) / train_labels[:, i].sum()
    label_weights.append(positive_weight)

class_weights = torch.tensor(label_weights, dtype=torch.float)
print("Class weights:", class_weights)

print("Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


print("Preparing datasets...")
train_dataset = EmotionDataset(train_df['text'].tolist(), train_labels, tokenizer)
dev_dataset = EmotionDataset(dev_df['text'].tolist(), dev_labels, tokenizer)

print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(emotions),
    problem_type="multi_label_classification"
)


def custom_loss(outputs, labels):
    logits = outputs.logits
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(logits.device))
    return loss_fn(logits, labels.float())


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)      # Forward pass
        loss = custom_loss(outputs, labels)  # Compute custom loss
        return (loss, outputs) if return_outputs else loss


print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

print("Initializing Trainer...")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Starting training...")
trainer.train()
print("Training complete!")


print("Saving the trained model...")
trainer.save_model("./results/final_model")


print("Evaluating on validation data...")
results = trainer.evaluate()
print("Evaluation Results:", results)
