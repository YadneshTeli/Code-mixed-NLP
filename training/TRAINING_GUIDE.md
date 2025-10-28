# 🎓 Model Training Guide

Complete guide to training HingBERT and CM-BERT models for production use.

---

## 📊 Dataset Overview

**Source:** LinCE (Linguistic Code-switching Evaluation) Benchmark  
**Location:** `research files/archive/`  
**URL:** https://www.kaggle.com/datasets/thedevastator/unlock-universal-language-with-the-lince-dataset

### Available Datasets

| Task | Language Pair | Train | Val | Test | Total |
|------|---------------|-------|-----|------|-------|
| **LID** (Language ID) | Hindi-English | 1.4 MB | 234 KB | 378 KB | ~2 MB |
| **LID** | Spanish-English | 4.1 MB | 646 KB | 1.1 MB | ~6 MB |
| **LID** | MSA-Egyptian Arabic | 3.4 MB | 439 KB | 512 KB | ~4 MB |
| **LID** | Nepali-English | 1.9 MB | 308 KB | 514 KB | ~3 MB |
| **SA** (Sentiment) | Spanish-English | 3.1 MB | 477 KB | 1.2 MB | ~5 MB |
| **NER** (Named Entities) | Hindi-English | 404 KB | 103 KB | 155 KB | ~662 KB |
| **POS** (Part-of-Speech) | Hindi-English | 462 KB | 69 KB | 109 KB | ~640 KB |

---

## 🎯 Training Objectives

### 1. HingBERT for Language Identification
**Goal:** Replace untrained L3Cube model with fully trained version

**Current Issue:**
```
⚠️ Some weights of BertModel were not initialized from checkpoint
['pooler.dense.bias', 'pooler.dense.weight']
```

**Solution:** Train from `bert-base-multilingual-cased` on Hindi-English LID data

**Expected Performance:**
- Token-level accuracy: >95% (state-of-the-art)
- F1 Score (macro): >0.92
- Training time: 30-60 min (CPU), 5-10 min (GPU)

**Output:** Token-level labels
- `lang1` - English token
- `lang2` - Hindi token
- `ne` - Named entity
- `other` - Punctuation, etc.

---

### 2. CM-BERT for Sentiment Analysis
**Goal:** Replace untrained L3Cube model with fully trained version

**Current Issue:**
```
❌ Returns 'LABEL_1' instead of 'positive'/'negative'/'neutral'
⚠️ Classifier head not initialized
```

**Solution:** Train from `xlm-roberta-base` on sentiment analysis data

**Expected Performance:**
- Accuracy: >80% on code-mixed text
- F1 Score (weighted): >0.78
- Training time: 45-90 min (CPU), 10-15 min (GPU)

**Output:** Sentence-level sentiment
- `positive` (label_2)
- `negative` (label_0)
- `neutral` (label_1)

---

## 🚀 Quick Start

### Step 1: Install Training Dependencies

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install training packages
pip install -r training/requirements-training.txt
```

**Additional Packages Installed:**
- `scikit-learn` - Metrics (accuracy, F1, classification report)
- `datasets` - HuggingFace datasets library
- `accelerate` - Faster training
- `evaluate` - Additional metrics
- `tensorboard` - Training visualization (optional)

### Step 2: Verify Dataset

```bash
# Check dataset files exist
python -c "
from pathlib import Path
data_dir = Path('research files/archive')
files = ['lid_hineng_train.csv', 'sa_spaeng_train.csv']
for f in files:
    path = data_dir / f
    if path.exists():
        print(f'✅ {f} ({path.stat().st_size / 1024:.0f} KB)')
    else:
        print(f'❌ {f} MISSING')
"
```

### Step 3: Run Training

```bash
# Interactive training menu
python train_models.py

# Or train specific models directly:
python training/train_hingbert_lid.py      # HingBERT only
python training/train_cmbert_sentiment.py  # CM-BERT only
```

---

## 📁 Training Scripts

### 1. `train_hingbert_lid.py` - HingBERT Language ID

**Input Format (LinCE CSV):**
```csv
idx,words,lid
0,['Good' 'vibe' 'tribe' '.'],['lang1' 'lang1' 'lang1' 'other']
1,['I' 'love' 'coding' 'बहुत' 'अच्छा' 'है'],['lang1' 'lang1' 'lang1' 'lang2' 'lang2' 'lang2']
```

**Training Process:**
1. Load LinCE Hindi-English LID dataset
2. Parse token-label pairs from CSV
3. Tokenize with `bert-base-multilingual-cased`
4. Align labels with subword tokens
5. Train for 5 epochs with early stopping
6. Evaluate on test set
7. Save to `models/trained/hingbert-lid-hinglish/`

**Hyperparameters:**
```python
learning_rate = 2e-5
batch_size = 16
epochs = 5
max_length = 128
warmup_steps = 500
weight_decay = 0.01
```

**Output Files:**
- `config.json` - Model configuration
- `pytorch_model.bin` - Trained weights
- `tokenizer_config.json` - Tokenizer settings
- `vocab.txt` - Vocabulary file

---

### 2. `train_cmbert_sentiment.py` - CM-BERT Sentiment

**Input Format (LinCE CSV):**
```csv
idx,words,sentiment
0,['This' 'is' 'amazing' '!'],'positive'
1,['Terrible' 'movie' 'worst' 'ever'],'negative'
```

**Training Process:**
1. Load Spanish-English sentiment dataset (transferable to Hinglish)
2. Parse text and sentiment labels
3. Tokenize with `xlm-roberta-base`
4. Train sequence classification head
5. Evaluate on test set
6. Save to `models/trained/cmbert-sentiment-codemixed/`

**Hyperparameters:**
```python
learning_rate = 2e-5
batch_size = 32
epochs = 5
max_length = 128
warmup_ratio = 0.1
weight_decay = 0.01
```

**Label Mapping:**
```python
{
    "positive": 2,   # High sentiment
    "negative": 0,   # Low sentiment
    "neutral": 1     # Neutral sentiment
}
```

---

## 💻 Training Environment

### Minimum Requirements
- **CPU:** 4+ cores
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** 5 GB free space
- **Time:** 1-2 hours for both models (CPU)

### Recommended (GPU)
- **GPU:** NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3060)
- **CUDA:** 11.8 or later
- **Time:** 15-25 minutes for both models

### Using GPU (Optional)

```bash
# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Training scripts will automatically use GPU if available
```

---

## 📊 Expected Training Output

### HingBERT LID Training Log

```
======================================================================
🔧 TRAINING HINGBERT FOR HINGLISH LID (Token-Level)
======================================================================

📊 Label Mapping:
  lang1      -> 0
  lang2      -> 1
  ne         -> 2
  other      -> 3

📁 Loading datasets from: ./research files/archive
  Loaded 10000 samples from lid_hineng_train.csv
  Loaded 1500 samples from lid_hineng_validation.csv
  Loaded 2500 samples from lid_hineng_test.csv

✅ Dataset loaded:
  Train: 10000 samples
  Validation: 1500 samples
  Test: 2500 samples

🤖 Loading base model: bert-base-multilingual-cased
✅ Model initialized with 4 labels

======================================================================
🚀 STARTING TRAINING
======================================================================
Epoch 1/5: [████████████████████] 100% | Loss: 0.234 | Acc: 0.912
Epoch 2/5: [████████████████████] 100% | Loss: 0.156 | Acc: 0.941
Epoch 3/5: [████████████████████] 100% | Loss: 0.098 | Acc: 0.967
Epoch 4/5: [████████████████████] 100% | Loss: 0.067 | Acc: 0.978
Epoch 5/5: [████████████████████] 100% | Loss: 0.051 | Acc: 0.983

======================================================================
📊 EVALUATING ON TEST SET
======================================================================
Test Results:
  accuracy: 0.9756
  f1_macro: 0.9234
  f1_weighted: 0.9689

✅ HINGBERT LID TRAINING COMPLETE!
```

### CM-BERT Sentiment Training Log

```
======================================================================
🔧 TRAINING CM-BERT FOR CODE-MIXED SENTIMENT ANALYSIS
======================================================================

📊 Sentiment Label Mapping:
  positive   -> 2
  negative   -> 0
  neutral    -> 1

📁 Loading sentiment datasets from: ./research files/archive
  Loaded 8500 samples from sa_spaeng_train.csv
  Label distribution: {'positive': 3200, 'negative': 3100, 'neutral': 2200}

======================================================================
🚀 STARTING TRAINING
======================================================================
Epoch 1/5: [████████████████████] 100% | Loss: 0.812 | Acc: 0.645
Epoch 2/5: [████████████████████] 100% | Loss: 0.567 | Acc: 0.743
Epoch 3/5: [████████████████████] 100% | Loss: 0.423 | Acc: 0.812
Epoch 4/5: [████████████████████] 100% | Loss: 0.334 | Acc: 0.856
Epoch 5/5: [████████████████████] 100% | Loss: 0.278 | Acc: 0.879

======================================================================
📊 EVALUATING ON TEST SET
======================================================================
Test Results:
  accuracy: 0.8234
  f1_macro: 0.7912
  f1_weighted: 0.8156
  f1_positive: 0.8567
  f1_negative: 0.8234
  f1_neutral: 0.6934

✅ CM-BERT SENTIMENT TRAINING COMPLETE!
```

---

## 🔧 Post-Training Integration

### Step 1: Update HingBERT Detector

Edit `app/language_detection/hingbert_detector.py`:

```python
# Change from:
MODEL_NAME = "l3cube-pune/hing-bert"

# To:
MODEL_NAME = "./models/trained/hingbert-lid-hinglish"
```

### Step 2: Update CM-BERT Analyzer

Edit `app/sentiment_analysis/cmbert_analyzer.py`:

```python
# Change from:
model_name = "l3cube-pune/hing-roberta"

# To:
model_name = "./models/trained/cmbert-sentiment-codemixed"
```

### Step 3: Test Updated Models

```bash
# Run all tests
pytest app/tests/test_api_integration.py app/tests/test_v2_new_endpoints.py -v

# Expected: 46/46 tests passing ✅
```

### Step 4: Verify Inference

```bash
# Test HingBERT
python -c "
from app.language_detection.hingbert_detector import HingBERTDetector
detector = HingBERTDetector()
result = detector.detect('I love coding यह बहुत अच्छा है')
print(result)
"

# Test CM-BERT
python -c "
from app.sentiment_analysis.cmbert_analyzer import CMBERTAnalyzer
analyzer = CMBERTAnalyzer()
result = analyzer.analyze('This movie is absolutely amazing!')
print(result)
"
```

---

## 📈 Monitoring Training

### TensorBoard (Optional)

```bash
# Start TensorBoard
tensorboard --logdir=models/trained/

# Open browser to: http://localhost:6006

# View:
# - Training/validation loss curves
# - Accuracy over epochs
# - Learning rate schedule
# - Gradient norms
```

### Weights & Biases (Optional)

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Training scripts will automatically log to W&B
# View at: https://wandb.ai
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```python
# Reduce batch size in training script
per_device_train_batch_size = 8  # Instead of 16 or 32

# Or enable gradient accumulation
gradient_accumulation_steps = 2  # Effective batch size = 8 * 2 = 16
```

### Issue: Training Too Slow

**Solutions:**
1. **Use GPU** (20x faster)
2. **Reduce epochs** (3 instead of 5)
3. **Reduce max_length** (64 instead of 128)
4. **Use smaller batch size** with gradient accumulation

### Issue: Low Accuracy

**Solutions:**
1. **Train longer** (10 epochs instead of 5)
2. **Try different learning rates** (1e-5, 3e-5, 5e-5)
3. **Add more training data**
4. **Use data augmentation**

### Issue: Label Mismatch

```python
# Verify label mapping matches your data
LABEL_MAP = {
    "positive": 2,
    "negative": 0,
    "neutral": 1
}

# Check dataset labels
import pandas as pd
df = pd.read_csv('research files/archive/sa_spaeng_train.csv')
print(df['sentiment'].unique())
```

---

## 📚 Additional Resources

### Papers
- **LinCE Benchmark:** "LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation" (2020)
- **HingBERT:** "L3Cube-HingBERT: Hindi-English Code-Mixed BERT" (2021)
- **XLM-RoBERTa:** "Unsupervised Cross-lingual Representation Learning at Scale" (2019)

### Datasets
- **LinCE:** https://ritual.uh.edu/lince/
- **Kaggle:** https://www.kaggle.com/datasets/thedevastator/unlock-universal-language-with-the-lince-dataset

### Documentation
- **Transformers:** https://huggingface.co/docs/transformers
- **Datasets:** https://huggingface.co/docs/datasets
- **Training Tips:** https://huggingface.co/docs/transformers/training

---

## ✅ Success Criteria

**Training Successful When:**
- ✅ Training completes without errors
- ✅ Test accuracy > 90% (LID) or > 75% (Sentiment)
- ✅ F1 scores consistent across classes
- ✅ Model saves to output directory
- ✅ Inference works on sample inputs
- ✅ All API tests pass (46/46)

**Ready for Production When:**
- ✅ Models integrated into app
- ✅ All tests passing
- ✅ Response format matches API contract
- ✅ Labels are human-readable (not LABEL_1)
- ✅ Performance acceptable (<2s response time)

---

**Generated:** 2025-10-28  
**Dataset:** LinCE (Linguistic Code-switching Evaluation)  
**Models:** HingBERT (LID) + CM-BERT (Sentiment)  
**Training Time:** 1-2 hours (CPU), 15-25 minutes (GPU)
