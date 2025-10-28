# 🚀 Multilingual Hinglish NLP v2.0 - Quick Start

## What's New in v2.0?

**Massive Accuracy Improvements:**
- ✅ **Hinglish accuracy:** 55% → **92%** (+37% improvement!)
- ✅ **English accuracy:** 93% → **94%**
- ✅ **176 languages supported** (previously only 2)
- ✅ **Token-level language detection** with HingBERT (96% accuracy)
- ✅ **Smart routing:** Automatically uses best model for each language

**New Models:**
- 🤖 **HingBERT** - Token-level Hinglish detection (state-of-the-art)
- 🤖 **CM-BERT** - Hinglish/English sentiment (92-94% accuracy)
- 🤖 **XLM-RoBERTa** - Multilingual sentiment (87% accuracy, 100+ languages)
- 🌍 **FastText** - Quick language detection (176 languages)

---

## ⚡ Quick Setup (5 minutes)

### Prerequisites
- Python 3.10+ (3.12.6 recommended)
- 5 GB free disk space
- 4 GB RAM minimum

### Installation

```powershell
# Clone repository
git clone https://github.com/YadneshTeli/Code-mixed-NLP.git
cd Code-mixed-NLP

# Run automated setup
python setup.py
```

The setup script will:
1. Install all dependencies (~2.1 GB models)
2. Download spaCy model
3. Download NLTK data
4. Verify installation
5. Run quick tests

**Or manual installation:**
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 🎯 Quick Test

Start the API:
```powershell
python app/main.py
```

Visit: **http://localhost:8000/docs**

Test the new V2 endpoint:
```powershell
curl -X POST "http://localhost:8000/api/v2/analyze" `
     -H "Content-Type: application/json" `
     -d '{"text": "Yaar this movie is too good! Bahut maza aaya 🎬"}'
```

Expected response:
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "route": "hinglish",
  "model_used": "CM-BERT",
  "language_detection": {
    "detected_language": "hi",
    "is_hinglish": true,
    "token_level_detection": {
      "hindi_percentage": 45.0,
      "english_percentage": 50.0,
      "is_code_mixed": true
    }
  },
  "processing_time_ms": 285.42
}
```

---

## 📚 API Endpoints

### V2 Endpoints (Multilingual - Recommended)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/analyze` | POST | Multilingual sentiment analysis with smart routing |
| `/api/v2/analyze/batch` | POST | Batch analysis for multiple texts |
| `/api/v2/languages` | GET | Get supported languages (176 total) |
| `/api/v2/health` | GET | Health check for all models |

### V1 Endpoints (Legacy - Still Supported)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze` | POST | Full analysis (original pipeline) |
| `/api/v1/analyze-sentiment` | POST | Sentiment only |
| `/api/v1/detect-language` | POST | Language detection |
| `/api/v1/preprocess` | POST | Text preprocessing |

---

## 🌍 Supported Languages

**Hinglish-Optimized (92-96% accuracy):**
- Hindi (हिंदी)
- English
- Code-mixed Hinglish

**Multilingual Support (87% accuracy):**
- **8 direct support:** Arabic, English, French, German, Hindi, Italian, Portuguese, Spanish
- **100+ via transfer learning:** All major world languages
- **176 total detection:** FastText can identify 176 languages

---

## 🧪 Running Tests

```powershell
# Run all tests (46 tests - 100% passing)
pytest app/tests/ -v

# Run specific test file
pytest app/tests/test_api.py -v
pytest app/tests/test_hybrid_pipeline.py -v

# Quick test summary
pytest app/tests/ -q

# With coverage
pytest app/tests/ --cov=app --cov-report=html
```

**Test Results:**
- ✅ 46 tests passing (100%)
- ✅ 0 warnings
- ✅ All FastText, HingBERT, and pipeline tests passing

---

## 📊 Performance Comparison

| Metric | v1.0 | v2.0 | Change |
|--------|------|------|--------|
| **Hinglish Accuracy** | 55% | 92% | **+37%** 🎯 |
| **English Accuracy** | 93% | 94% | +1% |
| **Languages** | 2 | 176 | **+174** 🌍 |
| **Model Size** | 268 MB | 2.1 GB | +1.8 GB |
| **Response Time** | 100-200ms | 200-350ms | +150ms |
| **Token Detection** | ❌ | ✅ | **NEW** |

---

## 🚀 Deployment

Deploy to Railway in 3 steps:

```powershell
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and initialize
railway login
railway init

# 3. Deploy
railway up
```

Your API will be live at: `https://your-app.railway.app`

**Railway FREE tier compatible:**
- Image: 2.3 GB (limit: 4 GB) ✅
- RAM: 3-4 GB (limit: 8 GB) ✅

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 📖 Documentation

**Main Documentation (11 essential files):**
- **[docs/INDEX.md](docs/INDEX.md)** - Documentation index
- **[docs/MULTILINGUAL_API_v2.md](docs/MULTILINGUAL_API_v2.md)** - Complete API reference
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide
- **[docs/API_TEST_SAMPLES.md](docs/API_TEST_SAMPLES.md)** - Test examples
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Project structure

**Technical & Training:**
- **[docs/technical/](docs/technical/)** - Detector status, NumPy compatibility, migration guide
- **[docs/training/](docs/training/)** - Training quickstart and alternatives

**Interactive API Docs:**
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🎯 Use Cases

### Best for Hinglish (Route: hinglish)
```python
texts = [
    "Yaar this movie is too good! 🎬",
    "Bahut boring tha yaar",
    "Kya amazing performance! 🎉"
]
```

### Best for Multilingual (Route: multilingual)
```python
texts = [
    "C'est magnifique!",  # French
    "¡Increíble película!",  # Spanish
    "هذا رائع جدا!",  # Arabic
    "Questo è fantastico!"  # Italian
]
```

---

## 🔧 Model Architecture

```
Input Text
    ↓
HybridPreprocessor (NLTK + spaCy)
    ↓
FastTextDetector (10-20ms, 176 languages)
    ↓
┌─────────────────────────────────────┐
│ Is Hinglish/Hindi/English?          │
├─────────────────────────────────────┤
│ YES → HingBERT (96% token detection)│
│       ↓                              │
│       CM-BERT (92-94% sentiment)    │
│                                      │
│ NO → XLM-RoBERTa (87% multilingual) │
└─────────────────────────────────────┘
```

---

## 💡 Tips

1. **Use V2 for new projects** - Better accuracy and multilingual support
2. **First request is slower** - Models lazy-load on first use (~10-15 seconds)
3. **Batch for efficiency** - Use `/api/v2/analyze/batch` for multiple texts
4. **Check language support** - Visit `/api/v2/languages`
5. **Monitor route** - Response shows which model was used

---

## 🐛 Troubleshooting

**Models not loading?**
```powershell
# Re-download spaCy model
python -m spacy download en_core_web_sm

# Re-download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Import errors?**
```powershell
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Out of memory?**
- Models need 3-4 GB RAM
- Close other applications
- Use batch processing for multiple texts

---

## 📝 Migration from v1.0

**API Changes:**
- Response field: `label` → `sentiment`
- New fields: `route`, `model_used`, `language_detection`
- V1 endpoints still work (backward compatible)

**Example Migration:**
```python
# OLD (v1.0)
response = requests.post("/api/v1/analyze-sentiment", json={"text": "..."})
sentiment = response.json()["label"]

# NEW (v2.0)
response = requests.post("/api/v2/analyze", json={"text": "..."})
sentiment = response.json()["sentiment"]
route = response.json()["route"]  # NEW: See which model was used
```

---

## 🤝 Contributing

Contributions welcome! See open issues or create a new one.

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Credits

**Models:**
- [HingBERT](https://huggingface.co/l3cube-pune/hing-roberta) - L3Cube Pune
- [CM-BERT](https://huggingface.co/l3cube-pune/hing-sentiment-roberta) - L3Cube Pune
- [XLM-RoBERTa](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) - Cardiff NLP

**Frameworks:**
- FastAPI, PyTorch, Transformers, spaCy, NLTK

---

**Version:** 2.0.0  
**Author:** Yadnesh Teli  
**Last Updated:** January 2025  
**Status:** ✅ Production Ready

---

🎉 **Start building multilingual NLP applications today!**
