import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from scipy.special import softmax

# --- 1. Check for CUDA (GPU) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 1. Using device: {device} ---")
if device == "cpu":
    print("--- WARNING: CUDA not available. Training will run on CPU and will be very slow. ---")
print("---------------------------------")
# --- End CUDA Check ---

MODEL_NAME = "microsoft/deberta-v3-base"
FINAL_MODEL_DIR = "final_semantic_model"
REPORT_DIR = "evaluation_report"
CHECKPOINT_DIR = "training_checkpoints"

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("--- 2. Loading and splitting dataset ---")
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("Error: dataset.csv not found.")
    print("Please make sure the file is in the same directory as this script.")
    df = pd.DataFrame(columns=['ext_type', 'text'])
    exit()

df.rename(columns={"ext_type": "label"}, inplace=True)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
df.dropna(subset=['label', 'text'], inplace=True)
df['label'] = df['label'].astype(int)

if len(df['label'].unique()) < 2:
    print("Error: The dataset must contain both 'ham' (0) and 'spam' (1) labels.")
    print(f"Found labels: {df['label'].unique()}")
    print("Please update dataset.csv with examples for both classes.")
    exit()

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Total examples: {len(df)}")
print(f"Training examples: {len(train_df)}")
print(f"Validation examples: {len(val_df)}")
print(f"Test examples: {len(test_df)}")
print("---------------------------------")


print("--- 3. Loading model and tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,
    use_safetensors=True  # Use secure safetensors format to avoid torch.load error
)
print("---------------------------------")


class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PhishingDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = PhishingDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = PhishingDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


print("--- 4. Starting model training ---")
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    no_cuda=(device == "cpu"), 
    save_safetensors=True # This saves new checkpoints securely
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# This logic automatically detects if a checkpoint exists
last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
if last_checkpoint:
    print(f"--- Resuming training from: {last_checkpoint} ---")
else:
    print("--- No checkpoint found. Starting training from scratch. ---")

# Pass the found checkpoint (or None) to the trainer
trainer.train(resume_from_checkpoint=last_checkpoint)

print("--- Training finished ---")
print("---------------------------------")


print(f"--- 5. Saving best model to {FINAL_MODEL_DIR} ---")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print("--- Model saved ---")
print("---------------------------------")


print(f"--- 6. Generating report on TEST set ---")
model_for_eval = AutoModelForSequenceClassification.from_pretrained(
    FINAL_MODEL_DIR,
    use_safetensors=True
)
eval_tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)

eval_trainer = Trainer(model=model_for_eval, args=training_args)

predictions = eval_trainer.predict(test_dataset)

y_true = predictions.label_ids
y_pred_logits = predictions.predictions
y_pred_probs = softmax(y_pred_logits, axis=1)
y_pred_labels = np.argmax(y_pred_logits, axis=1)

print("--- Generating Classification Report ---")
report = classification_report(y_true, y_pred_labels, target_names=["Ham (0)", "Phishing (1)"])
report_path = os.path.join(REPORT_DIR, "classification_report.txt")

with open(report_path, "w") as f:
    f.write("--- Semantic Model Classification Report ---\n\n")
    f.write(report)

print(report)
print(f"Classification report saved to {report_path}")

print("--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred_labels)
cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham (0)", "Phishing (1)"],
            yticklabels=["Ham (0)", "Phishing (1)"])
plt.title("Confusion Matrix for Semantic Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

print("--- Generating Probability Scatterplot ---")
prob_df = pd.DataFrame({
    'true_label': y_true,
    'predicted_phishing_prob': y_pred_probs[:, 1]
})
prob_path = os.path.join(REPORT_DIR, "probability_scatterplot.png")

plt.figure(figsize=(10, 6))
sns.stripplot(data=prob_df, x='true_label', y='predicted_phishing_prob', jitter=0.2, alpha=0.7)
plt.title("Model Confidence: Predicted Phishing Probability vs. True Label")
plt.xlabel("True Label")
plt.ylabel("Predicted Phishing Probability")
plt.xticks([0, 1], ["Ham (0)", "Phishing (1)"])
plt.axhline(0.5, color='r', linestyle='--', label='Decision Boundary (0.5)')
plt.legend()
plt.savefig(prob_path)
plt.close()
print(f"Probability scatterplot saved to {prob_path}")

print("---------------------------------")
print(f"--- Evaluation Complete. Reports saved to {REPORT_DIR} ---")