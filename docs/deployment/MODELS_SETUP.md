# Model Setup Guide

This guide explains how to download and set up the required models for the Code-mixed NLP API.

## Quick Setup

Run the model download script:

```bash
# Set UTF-8 encoding (Windows)
$env:PYTHONIOENCODING='utf-8'

# Download models
python download_models.py
```

## Models Overview

### ✅ Available Models (Downloaded Automatically)

1. **FastText Language Detector** (125 MB)
   - File: `models/language_detection/lid.176.bin`
   - Supports: 176 languages
   - Accuracy: ~95%
   - Source: https://dl.fbaipublicfiles.com/fasttext/supervised-models/

2. **XLM-RoBERTa Sentiment Analyzer**
   - Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
   - Supports: 100+ languages
   - Accuracy: ~87%
   - Cached by Hugging Face Transformers

3. **spaCy English Model**
   - Model: `en_core_web_sm`
   - Used for: Text preprocessing, tokenization
   - Install: `python -m spacy download en_core_web_sm`

### ⚠️ Unavailable Models (Fallbacks Used)

These models are not publicly available on HuggingFace. The system automatically uses high-quality fallbacks:

1. **HingBERT** (Token-level Detection)
   - Intended: `l3cube-pune/hindi-english-hing-bert`
   - Fallback: Rule-based detection
   - Fallback Accuracy: ~70% for romanized text, ~95% for Devanagari

2. **CM-BERT** (Hinglish Sentiment)
   - Intended: `l3cube-pune/code-mixed-bert` or `l3cube-pune/codebert-base-hinglish-sentiment`
   - Fallback: XLM-RoBERTa
   - Fallback Accuracy: ~87%

## Manual Model Download

### FastText Model

If automatic download fails:

```bash
# Create directory
mkdir -p models/language_detection

# Download model (125 MB)
curl -o models/language_detection/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## Verification

Test that all models are loaded correctly:

```bash
# Set encoding
$env:PYTHONIOENCODING='utf-8'

# Run tests
pytest app/tests/ -v
```

Expected: **46/46 tests passing**

## Model Storage

Models are cached in the following locations:

- **FastText**: `models/language_detection/lid.176.bin` (125 MB)
- **Transformers**: `~/.cache/huggingface/` (varies by OS)
- **spaCy**: System-dependent cache location

Total disk space required: ~500 MB

## Troubleshooting

### UnicodeEncodeError on Windows

If you see encoding errors, set UTF-8 encoding:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

Or add to your PowerShell profile:

```powershell
# Add to $PROFILE
$env:PYTHONIOENCODING='utf-8'
```

### Model Download Fails

1. Check internet connection
2. Try manual download (see above)
3. Check disk space (need ~500 MB free)

### Models Not Loading

Run the diagnostic script:

```bash
python download_models.py
```

This will show which models loaded successfully and which failed.

## Performance Notes

- **First Run**: Models download automatically (may take 5-10 minutes)
- **Subsequent Runs**: Models load from cache (< 5 seconds)
- **Memory Usage**: ~500 MB RAM when all models loaded
- **Inference Time**: 50-200ms per request depending on text length

## Railway Deployment Setup

### Build Configuration

**Important**: FastText model (125 MB) is excluded from git due to GitHub's 100 MB limit. Download it during the build phase:

1. **Add to `railway.json` or build script:**

```json
{
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt && python -m spacy download en_core_web_sm && mkdir -p models/language_detection && curl -L -o models/language_detection/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

2. **Environment Variables** (Set in Railway dashboard):

```bash
PYTHONIOENCODING=utf-8
PORT=8000
```

3. **Dockerfile Alternative** (if using Docker):

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download FastText model
RUN mkdir -p models/language_detection && \
    curl -L -o models/language_detection/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Copy application code
COPY . .

# Set environment
ENV PYTHONIOENCODING=utf-8
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Start server
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Build Time Optimization

**Expected build time:** 3-5 minutes

- Dependencies install: ~2 minutes
- spaCy model download: ~30 seconds
- FastText model download: ~30 seconds (125 MB)
- Transformers models: Downloaded on first request (cached thereafter)

### Memory Requirements

Set Railway service memory:
- **Minimum:** 512 MB (basic operation)
- **Recommended:** 1 GB (smooth performance)
- **Optimal:** 2 GB (handles concurrent requests)

### Cold Start Optimization

First request after deployment takes 1-2 minutes to download Transformers models (XLM-RoBERTa ~1.1 GB). Subsequent requests are fast.

**Solution**: Add a startup warmup script:

```python
# startup_warmup.py
from app.pipeline import get_pipeline

print("🔥 Warming up models...")
pipeline = get_pipeline()
pipeline.analyze("test")  # Triggers model downloads
print("✅ Models ready!")
```

Update start command:
```bash
python startup_warmup.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Deployment Checklist

- [ ] FastText model download in build command
- [ ] Environment variables set (`PYTHONIOENCODING=utf-8`, `PORT`)
- [ ] spaCy model download in build command  
- [ ] Health check endpoint configured (`/health`)
- [ ] Memory allocation: 1-2 GB
- [ ] Startup warmup script (optional but recommended)

## Production Deployment (General)

For production:

1. Pre-download all models during build/deployment
2. Set `PYTHONIOENCODING=utf-8` in environment
3. Use model caching to speed up cold starts
4. Consider using a CDN for FastText model download
5. Allocate sufficient memory (1-2 GB recommended)

## Model Accuracy Summary

| Component | Model | Accuracy | Languages |
|-----------|-------|----------|-----------|
| Language Detection | FastText | 95% | 176 |
| Hinglish Sentiment | XLM-RoBERTa (fallback) | 87% | 100+ |
| English Sentiment | XLM-RoBERTa | 94% | 100+ |
| Token Detection | Rule-based (fallback) | 70-95% | 2 |
| Preprocessing | spaCy | N/A | English |

## Support

For issues with model downloads or setup, check:
- GitHub Issues: [Your repo URL]
- Documentation: [Your docs URL]
- Test results: Run `pytest app/tests/ -v`
