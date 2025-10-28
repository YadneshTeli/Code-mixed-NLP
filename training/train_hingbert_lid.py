"""
Train HingBERT for Hinglish Language Identification (Token-Level)
Uses LinCE dataset: Hindi-English code-switching corpus

Dataset: research files/archive/lid_hineng_*.csv
Labels: lang1 (English), lang2 (Hindi), ne (Named Entity), other
"""

import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
import ast
import os

print("=" * 70)
print("🔧 TRAINING HINGBERT FOR HINGLISH LID (Token-Level)")
print("=" * 70)

# Configuration
BASE_MODEL = "bert-base-multilingual-cased"  # Use this as base for HingBERT
OUTPUT_DIR = "./models/trained/hingbert-lid-hinglish"
DATA_DIR = "./research files/archive"

# Label mapping
LABEL_MAP = {
    "lang1": 0,  # English
    "lang2": 1,  # Hindi
    "ne": 2,     # Named Entity
    "other": 3,  # Punctuation, etc.
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

print(f"\n📊 Label Mapping:")
for label, idx in LABEL_MAP.items():
    print(f"  {label:10s} -> {idx}")

# Load datasets
print(f"\n📁 Loading datasets from: {DATA_DIR}")

def parse_lince_format(csv_path):
    """Parse LinCE CSV format with list-like strings"""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {os.path.basename(csv_path)}")
    
    examples = []
    for idx, row in df.iterrows():
        # Parse string representation of lists
        words = ast.literal_eval(row['words'])
        labels = ast.literal_eval(row['lid'])
        
        # Convert labels to IDs
        label_ids = [LABEL_MAP.get(label, LABEL_MAP['other']) for label in labels]
        
        examples.append({
            'tokens': words,
            'labels': label_ids
        })
    
    return examples

# Load train, validation, test
train_data = parse_lince_format(f"{DATA_DIR}/lid_hineng_train.csv")
val_data = parse_lince_format(f"{DATA_DIR}/lid_hineng_validation.csv")
test_data = parse_lince_format(f"{DATA_DIR}/lid_hineng_test.csv")

print(f"\n✅ Dataset loaded:")
print(f"  Train: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")

# Initialize tokenizer and model
print(f"\n🤖 Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABEL_MAP),
    id2label=ID2LABEL,
    label2id=LABEL_MAP
)

print(f"✅ Model initialized with {len(LABEL_MAP)} labels")

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    """Tokenize tokens and align labels with subword tokens"""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            # Special tokens get label -100
            if word_idx is None:
                label_ids.append(-100)
            # Only label first subword token of each word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Convert to datasets format
print("\n🔄 Tokenizing datasets...")
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

train_dataset = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=['tokens', 'labels']
)
val_dataset = val_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=['tokens', 'labels']
)
test_dataset = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=['tokens', 'labels']
)

print("✅ Tokenization complete")

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_labels = [[LABEL_MAP.get(p, p) for p, l in zip(pred, label) if l != -100]
                   for pred, label in zip(predictions, labels)]
    true_predictions = [[p for p, l in zip(pred, label) if l != -100]
                        for pred, label in zip(predictions, labels)]
    
    # Flatten for metrics
    flat_true = [item for sublist in true_labels for item in sublist]
    flat_pred = [item for sublist in true_predictions for item in sublist]
    
    accuracy = accuracy_score(flat_true, flat_pred)
    f1_macro = f1_score(flat_true, flat_pred, average='macro')
    f1_weighted = f1_score(flat_true, flat_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    push_to_hub=False,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
print("\n" + "=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print("=" * 70)

trainer.train()

# Evaluate on test set
print("\n" + "=" * 70)
print("📊 EVALUATING ON TEST SET")
print("=" * 70)

test_results = trainer.evaluate(test_dataset)
print(f"\nTest Results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# Save model
print(f"\n💾 Saving model to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")
print(f"🎯 Best F1 Score: {test_results.get('eval_f1_weighted', 'N/A'):.4f}")

# Test inference
print("\n" + "=" * 70)
print("🧪 TESTING INFERENCE")
print("=" * 70)

test_texts = [
    ["I", "love", "coding", "बहुत", "अच्छा", "है"],
    ["What", "time", "है", "?"],
    ["Python", "programming", "सीखना", "easy", "है"]
]

for tokens in test_texts:
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    
    predicted_labels = [ID2LABEL[p.item()] for p in predictions[0] 
                       if p.item() in ID2LABEL]
    
    print(f"\nTokens: {tokens}")
    print(f"Labels: {predicted_labels[:len(tokens)]}")

print("\n" + "=" * 70)
print("✅ HINGBERT LID TRAINING COMPLETE!")
print("=" * 70)
