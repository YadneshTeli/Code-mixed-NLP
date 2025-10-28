"""
Train CM-BERT for Hinglish Sentiment Analysis
Uses Spanish-English sentiment analysis dataset from LinCE as starting point
Then can be adapted for Hinglish

Dataset: research files/archive/sa_spaeng_*.csv
Labels: positive, negative, neutral
"""

import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
import ast
import os

print("=" * 70)
print("🔧 TRAINING CM-BERT FOR CODE-MIXED SENTIMENT ANALYSIS")
print("=" * 70)

# Configuration
BASE_MODEL = "xlm-roberta-base"  # Multilingual base for code-mixing
OUTPUT_DIR = "./models/trained/cmbert-sentiment-codemixed"
DATA_DIR = "./research files/archive"

# Label mapping for sentiment
LABEL_MAP = {
    "positive": 2,
    "negative": 0,
    "neutral": 1
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

print(f"\n📊 Sentiment Label Mapping:")
for label, idx in LABEL_MAP.items():
    print(f"  {label:10s} -> {idx}")

# Load datasets
print(f"\n📁 Loading sentiment datasets from: {DATA_DIR}")

def parse_sentiment_data(csv_path):
    """Parse LinCE sentiment CSV format"""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {os.path.basename(csv_path)}")
    
    examples = []
    label_counts = {label: 0 for label in LABEL_MAP.keys()}
    
    for idx, row in df.iterrows():
        # Parse words
        words = ast.literal_eval(row['words'])
        text = ' '.join(words)
        
        # Get sentiment label
        sentiment = row.get('sentiment', row.get('label', 'neutral'))
        
        # Map to label ID
        label_id = LABEL_MAP.get(sentiment.lower(), LABEL_MAP['neutral'])
        label_counts[ID2LABEL[label_id]] += 1
        
        examples.append({
            'text': text,
            'label': label_id
        })
    
    print(f"  Label distribution: {label_counts}")
    return examples

# Load train, validation, test for Spanish-English
# Note: We'll use this as initial training, then you can add Hinglish data
train_data = parse_sentiment_data(f"{DATA_DIR}/sa_spaeng_train.csv")
val_data = parse_sentiment_data(f"{DATA_DIR}/sa_spaeng_validation.csv")
test_data = parse_sentiment_data(f"{DATA_DIR}/sa_spaeng_test.csv")

print(f"\n✅ Dataset loaded:")
print(f"  Train: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")

# Initialize tokenizer and model
print(f"\n🤖 Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABEL_MAP),
    id2label=ID2LABEL,
    label2id=LABEL_MAP
)

print(f"✅ Model initialized with {len(LABEL_MAP)} sentiment classes")

# Tokenize function
def tokenize_function(examples):
    """Tokenize text inputs"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

# Convert to datasets format
print("\n🔄 Tokenizing datasets...")
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

print("✅ Tokenization complete")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    """Compute accuracy, F1, and per-class metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    # Per-class F1
    f1_per_class = f1_score(labels, predictions, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }
    
    # Add per-class F1 scores
    for idx, label_name in ID2LABEL.items():
        if idx < len(f1_per_class):
            metrics[f'f1_{label_name}'] = f1_per_class[idx]
    
    return metrics

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    push_to_hub=False,
    save_total_limit=2,
    fp16=False,  # Set to True if using GPU with FP16 support
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
    "This movie is absolutely amazing! I loved it!",
    "Terrible experience, waste of time and money.",
    "It was okay, nothing special.",
    "¡Excelente! Me encantó mucho.",
    "No me gustó para nada, muy malo."
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    
    print(f"\nText: {text}")
    print(f"Sentiment: {ID2LABEL[predicted_class]} (confidence: {confidence:.4f})")

print("\n" + "=" * 70)
print("✅ CM-BERT SENTIMENT TRAINING COMPLETE!")
print("=" * 70)
print("\n📝 Next Steps:")
print("  1. Add Hinglish sentiment data for fine-tuning")
print("  2. Update app/sentiment_analysis/cmbert_analyzer.py to use this model")
print("  3. Test with Hinglish examples")
print("  4. Deploy to production")
