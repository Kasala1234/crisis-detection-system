import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# 🔹 Load dataset
df = pd.read_csv("data/train.csv")
df = df[["text", "target"]]

# 🔥 Rename column (VERY IMPORTANT)
df = df.rename(columns={"target": "labels"})

# 🔥 Reduce dataset size (FASTER TRAINING)
df = df.sample(1000, random_state=42)

# 🔹 Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# 🔹 Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 🔹 Tokenization function
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128   # 🔥 reduce length for speed
    )

dataset = dataset.map(tokenize, batched=True)

# 🔹 Split dataset (IMPORTANT)
dataset = dataset.train_test_split(test_size=0.2)

# 🔹 Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 🔹 Training settings (OPTIMIZED)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_steps=200,              # 🔥 limit steps for speed
    logging_steps=20,
    save_steps=100,
    evaluation_strategy="steps",
    save_total_limit=1,
    logging_dir="./logs"
)

# 🔹 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# 🔥 Train
trainer.train()

# 🔹 Save model
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("✅ BERT training completed and saved!")