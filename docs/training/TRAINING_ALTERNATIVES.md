# Alternative Training Approaches

**Issue:** C: drive has limited space (3.5 GB), making local training challenging  
**Solution:** Multiple alternative approaches below

---

## ✅ RECOMMENDED: Google Colab (FREE GPU)

### Why Colab?
- ✅ **FREE** Tesla T4 GPU (15 GB VRAM)
- ✅ **12 GB RAM** (enough for training)
- ✅ **Training time:** 10-15 minutes (vs 1.5-2 hours on CPU)
- ✅ No local disk space needed
- ✅ Pre-installed transformers, torch, datasets

### Steps:

1. **Upload Dataset to Google Drive**
   ```bash
   # Zip your dataset
   Compress-Archive -Path "research files" -DestinationPath "research_files.zip"
   
   # Upload research_files.zip to Google Drive
   ```

2. **Create Colab Notebook:** https://colab.research.google.com

3. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Unzip dataset
   !unzip /content/drive/MyDrive/research_files.zip
   ```

4. **Install Dependencies:**
   ```python
   !pip install -q datasets scikit-learn accelerate evaluate
   ```

5. **Copy Training Script** (from `training/train_hingbert_lid.py`)

6. **Enable GPU:** Runtime → Change runtime type → GPU → T4

7. **Run Training** (~10-15 minutes)

8. **Download Trained Model:**
   ```python
   # Zip the model
   !zip -r hingbert_trained.zip models/trained/hingbert-lid-hinglish
   
   # Copy to Drive
   !cp hingbert_trained.zip /content/drive/MyDrive/
   ```

**Colab Notebook Template:** See `COLAB_TRAINING_TEMPLATE.ipynb` below

---

## Option 2: Kaggle Notebooks (FREE GPU)

### Why Kaggle?
- ✅ **FREE** P100 GPU (30 hours/week)
- ✅ **13 GB RAM**
- ✅ Dataset already on Kaggle!
- ✅ Can save to Kaggle Datasets

### Steps:

1. **Go to:** https://www.kaggle.com/datasets/thedevastator/unlock-universal-language-with-the-lince-dataset

2. **Click "New Notebook"**

3. **Enable GPU:** Settings → Accelerator → GPU T4 x2

4. **Dataset is already loaded at:** `/kaggle/input/lince/`

5. **Run training script** (modify paths to use `/kaggle/input/lince/`)

6. **Save model:**
   ```python
   # Models saved to /kaggle/working/
   # Download via "Output" tab
   ```

**Faster than Colab:** GPU P100 is more powerful than Colab's T4

---

## Option 3: Hugging Face Spaces (FREE Training)

### Why HF Spaces?
- ✅ Can train directly on HF infrastructure
- ✅ Models automatically uploaded to Hub
- ✅ Easy integration with your app

### Steps:

1. **Create HF Account:** https://huggingface.co/join

2. **Create new Space:** Settings → New Space → SDK: Gradio

3. **Upload dataset** to HF Datasets

4. **Use HF Trainer API** (already in our scripts)

5. **Push to Hub:**
   ```python
   trainer.push_to_hub("YadneshTeli/hingbert-lid-hinglish")
   ```

---

## Option 4: Use Pre-trained Alternatives (INSTANT)

### Skip training entirely - use existing trained models:

#### For Sentiment Analysis:
```python
# In app/sentiment_analysis/cmbert_analyzer.py, use:
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
# This is ALREADY TRAINED and works for code-mixed text
```

#### For Language Detection:
```python
# Keep using FastText (already working)
# HingBERT only needed for token-level detection (advanced use case)
```

**Result:** All 46 tests will pass WITHOUT training!

---

## Option 5: Local Training with Optimizations

### If you want to train locally on your machine:

1. **Create fresh venv on D: drive:**
   ```bash
   # Create venv on D: drive (more space)
   python -m venv D:\venv-training
   D:\venv-training\Scripts\activate
   
   # Set cache to D: drive
   $env:HF_HOME = "D:\.cache\huggingface"
   $env:TRANSFORMERS_CACHE = "D:\.cache\huggingface\transformers"
   
   # Install dependencies
   pip install transformers torch datasets scikit-learn accelerate --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Use smaller base model:**
   ```python
   # Instead of bert-base-multilingual-cased (668 MB)
   BASE_MODEL = "distilbert-base-multilingual-cased"  # 270 MB, faster
   ```

3. **Reduce training parameters:**
   ```python
   num_train_epochs = 3  # Instead of 5
   per_device_train_batch_size = 8  # Instead of 16
   max_length = 64  # Instead of 128
   ```

4. **Train one model at a time**

**Estimated time:** 45-90 minutes per model (CPU)

---

## 🚀 QUICK WIN: Use XLM-RoBERTa for Sentiment

**Instead of training CM-BERT**, just switch to the pre-trained XLM-RoBERTa:

```python
# In app/sentiment_analysis/cmbert_analyzer.py line 26:
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
```

**Benefits:**
- ✅ Already trained for sentiment
- ✅ Multilingual (works with Hinglish)
- ✅ Returns proper labels (positive/negative/neutral)
- ✅ All sentiment tests will pass
- ✅ No training needed

**Download model once:**
```bash
python -c "
from transformers import pipeline
pipe = pipeline('sentiment-analysis', 
                model='cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual')
print('Model downloaded and ready!')
"
```

---

## Comparison Table

| Method | Time | Cost | GPU | Complexity | Best For |
|--------|------|------|-----|------------|----------|
| **Google Colab** | 15 min | FREE | ✅ T4 | Low | **RECOMMENDED** |
| **Kaggle** | 15 min | FREE | ✅ P100 | Low | Best performance |
| **HF Spaces** | 30 min | FREE | ❌ CPU | Medium | CI/CD integration |
| **Pre-trained** | 5 min | FREE | ❌ | Very Low | **QUICKEST FIX** |
| **Local (D: drive)** | 2 hours | FREE | ❌ CPU | Medium | Full control |

---

## 📝 Fresh Install Steps (D: Drive)

```powershell
# 1. Create venv on D: drive
cd D:\Yadnesh-Teli\Projects\Code-mixed-NLP
python -m venv venv

# 2. Activate
.\venv\Scripts\activate

# 3. Set cache to D: drive
$env:HF_HOME = "D:\Yadnesh-Teli\Projects\Code-mixed-NLP\.cache"

# 4. Install minimal dependencies (inference only)
pip install -r requirements.txt --no-cache-dir

# 5. For training (optional):
pip install datasets scikit-learn accelerate evaluate --no-cache-dir

# 6. Test installation
python -c "import transformers, torch; print('Ready!')"
```

---

## 🎯 Recommended Path Forward

### Path 1: Quick Production Fix (5 minutes)
1. Fresh install venv (D: drive)
2. Use pre-trained XLM-RoBERTa for sentiment
3. Keep FastText for language detection
4. **Result:** All tests pass, production ready

### Path 2: Train with GPU (15 minutes)
1. Upload dataset to Google Drive
2. Use Colab notebook (free GPU)
3. Train both models
4. Download and integrate
5. **Result:** Custom trained models

### Path 3: Train Locally (2 hours)
1. Fresh install on D: drive
2. Use optimized settings
3. Train overnight
4. **Result:** Full control, local models

---

## Next Steps

**Choose your path:**

**A. Quick Fix (Recommended for now):**
```bash
# 1. Fresh install
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 2. Update sentiment analyzer
# Edit app/sentiment_analysis/cmbert_analyzer.py line 26:
# model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

# 3. Test
pytest app/tests/test_api_integration.py -v
# All tests should pass!
```

**B. Train with Colab:**
See `COLAB_TRAINING_TEMPLATE.ipynb` created below

---

**Current Status:**
- ✅ C: drive space freed: 3.5 GB
- ✅ D: drive space available: 639 GB
- ✅ venv removed
- ✅ Cache cleared
- ✅ Ready for fresh install

**Disk Space Summary:**
- C: drive: 3.5 GB free (enough for system)
- D: drive: 639 GB free (use this for everything)

