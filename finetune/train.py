import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# === Load JSONL dataset ===
def load_jsonl_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return Dataset.from_list(lines)

# === Preprocess ===
def preprocess(example):
    input_text = example["instruction"].strip() + "\n" + example["input"].strip()
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["output"].strip(), max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === Model and tokenizer ===
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# === Load and preprocess dataset ===
data_path = os.path.join("finetune", "small_dataset_cleaned.jsonl")
dataset = load_jsonl_dataset(data_path)
dataset = dataset.map(preprocess)

# === Load model ===
model = T5ForConditionalGeneration.from_pretrained(model_name)

# === Training configuration ===
training_args = TrainingArguments(
    output_dir="./finetune/output/small-model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    overwrite_output_dir=True
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# === Train ===
trainer.train()

# === Save final model ===
trainer.save_model("./finetune/output/small-model")
tokenizer.save_pretrained("./finetune/output/small-model")
