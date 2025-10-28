# 🌐 Multilingual Hinglish NLP Pipeline & REST API v2.0

A state-of-the-art Natural Language Processing (NLP) system for **multilingual text analysis** with specialized support for **Hinglish** (Hindi-English code-mixed) content. Features advanced language detection (176 languages), sentiment analysis, and a production-ready REST API with smart model routing.

## ✨ Features

### 🌍 Advanced Language Detection (v2.0)
- **176 languages** supported via FastText
- **Token-level detection** with HingBERT (96% accuracy)
- **Hinglish code-mixing detection** (92% accuracy)
- Fast detection (10-20ms per text)
- Confidence scoring and reliability indicators

### 🔧 Hybrid Text Preprocessing
- **spaCy + NLTK** combined approach
- Sentence segmentation and tokenization
- URL and mention removal
- Emoji preservation
- Devanagari script support
- Language-aware processing

### 😊 Multilingual Sentiment Analysis
- **Smart routing** based on detected language
- **CM-BERT** for Hinglish/English (92-94% accuracy)
- **XLM-RoBERTa** for 100+ languages (87% accuracy)
- 3-class sentiment (positive, neutral, negative)
- Confidence scores and detailed metrics
- Batch processing support

### 🚀 Production-Ready REST API
- **V1 & V2 endpoints** (backward compatible)
- FastAPI with auto-generated documentation
- Smart model routing for optimal accuracy
- Request/response validation with Pydantic
- CORS support for web applications
- Comprehensive error handling
- Health monitoring and status checks
- Swagger UI & ReDoc documentation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Model Information](#model-information)
- [Development](#development)

## 🔨 Installation

### Prerequisites
- Python 3.12+ (or 3.10+)
- pip package manager
- Virtual environment (recommended)
- ~4GB RAM for model loading

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Code-mixed-NLP
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Key Packages:**
- `fastapi==0.120.0` - Web framework
- `uvicorn==0.34.0` - ASGI server
- `pydantic==2.10.3` - Data validation
- `transformers==4.47.1` - NLP models
- `torch==2.5.1` - PyTorch
- `spacy==3.8.2` - Advanced NLP
- `fasttext-wheel==0.9.2` - Language detection
- `pytest==8.4.2` - Testing framework
- `httpx==0.28.1` - HTTP client

## 🚀 Quick Start

### 1. Start the API Server

**Option A: Using the batch file (Windows)**
```bash
start_server.bat
```

**Option B: Direct Python command**
```bash
python app/main.py
```

The server will start on `http://localhost:8000`

### 2. Access API Documentation

Open your browser and navigate to:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **API Info:** http://localhost:8000/

### 3. Test the API

**Using V2 endpoint (recommended):**
```bash
curl -X POST "http://localhost:8000/api/v2/analyze" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Yeh movie bahut accha hai! I loved it!\"}"
```

**Using V1 endpoint (legacy):**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Yeh movie bahut accha hai! I loved it!\"}"
```

**Using Python (V2 API):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v2/analyze",
    json={"text": "Yeh movie bahut accha hai! I loved it!"}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Route: {result['route']}")
print(f"Model: {result['model_used']}")
print(f"Hinglish: {result['language_detection']['is_hinglish']}")
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### API Versions

- **V2 API** (`/api/v2/*`) - Recommended, multilingual support with smart routing
- **V1 API** (`/api/v1/*`) - Legacy, maintained for backward compatibility

### V2 Endpoints (Recommended)

#### 1. Smart Multilingual Analysis (V2)
```http
POST /api/v2/analyze
```

**Request:**
```json
{
  "text": "Yaar this movie is too good! 🎬"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "confidence_level": "high",
  "scores": {
    "positive": 0.94,
    "negative": 0.03,
    "neutral": 0.03
  },
  "route": "hinglish",
  "model_used": "CM-BERT",
  "language_detection": {
    "detected_language": "hi",
    "confidence": 0.89,
    "is_hinglish": true,
    "is_reliable": true
  },
  "preprocessing": {
    "tokens_count": 7,
    "sentence_count": 1
  },
  "processing_time_ms": 285.42
}
```

#### 2. Batch Analysis (V2)
```http
POST /api/v2/analyze/batch
```

**Request:**
```json
{
  "texts": [
    "This is amazing!",
    "Yeh bahut accha hai!",
    "C'est magnifique!"
  ]
}
```

**Response:**
```json
{
  "count": 3,
  "results": [
    {
      "sentiment": "positive",
      "route": "hinglish",
      "model_used": "CM-BERT",
      ...
    },
    ...
  ]
}
```

#### 3. Supported Languages (V2)
```http
GET /api/v2/languages
```

**Response:**
```json
{
  "total_languages": 176,
  "hinglish_optimized": {
    "languages": ["hi", "en"],
    "model": "CM-BERT",
    "accuracy": "92-94%"
  },
  "multilingual_support": {
    "direct_support": ["ar", "en", "es", "fr", "de", "hi", "it", "pt"],
    "total_via_transfer": "100+",
    "model": "XLM-RoBERTa",
    "accuracy": "85-90%"
  }
}
```

#### 4. Health Check (V2)
```http
GET /api/v2/health
```

**Response:**
```json
{
  "pipeline": "healthy",
  "components": {
    "preprocessor": "healthy",
    "fasttext": "healthy",
    "hingbert": "healthy",
    "cmbert": "healthy",
    "xlm_roberta": "healthy"
  }
}
```

### V1 Endpoints (Legacy)

#### 5. Health Check (V1)
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

#### 6. Text Preprocessing (V1)
```http
POST /api/v1/preprocess
```

**Request:**
```json
{
  "text": "Check out https://example.com! 😊 #amazing"
}
```

**Response:**
```json
{
  "original": "Check out https://example.com! 😊 #amazing",
  "cleaned": "check out 😊 amazing",
  "tokens": ["check", "out", "😊", "amazing"],
  "token_count": 4
}
```

#### 7. Language Detection (V1)
```http
POST /api/v1/detect-language
```

**Request:**
```json
{
  "text": "Main bahut happy hoon today"
}
```

**Response:**
```json
{
  "tokens": ["main", "bahut", "happy", "hoon", "today"],
  "labels": ["lang2", "lang2", "lang1", "lang2", "lang1"],
  "statistics": {
    "lang1": {"count": 2, "percentage": 40.0},
    "lang2": {"count": 3, "percentage": 60.0}
  },
  "is_code_mixed": true,
  "dominant_language": "lang2"
}
```

**Language Labels:**
- `lang1` - English
- `lang2` - Hindi/Romanized Hindi
- `ne` - Named Entity
- `other` - Punctuation, numbers, special characters

#### 8. Sentiment Analysis (V1)
```http
POST /api/v1/analyze-sentiment
```

**Request:**
```json
{
  "text": "This is absolutely amazing!"
}
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 0.9998,
  "scores": {
    "positive": 0.9998,
    "negative": 0.0002
  }
}
```

#### 9. Full Analysis (V1)
```http
POST /api/v1/analyze
```

**Request:**
```json
{
  "text": "Yeh movie bahut accha hai! I loved it!"
}
```

**Response:**
```json
{
  "original_text": "Yeh movie bahut accha hai! I loved it!",
  "cleaned_text": "yeh movie bahut accha hai i loved it",
  "tokens": ["yeh", "movie", "bahut", "accha", "hai", "i", "loved", "it"],
  "token_count": 8,
  "language_detection": {
    "labels": ["lang2", "lang1", "lang2", "lang2", "lang2", "lang1", "lang1", "lang1"],
    "statistics": {
      "lang1": {"count": 4, "percentage": 50.0},
      "lang2": {"count": 4, "percentage": 50.0}
    },
    "is_code_mixed": true,
    "dominant_language": "lang1"
  },
  "sentiment": {
    "label": "positive",
    "confidence": 0.9998,
    "scores": {
      "positive": 0.9998,
      "negative": 0.0002
    }
  }
}
```

#### 10. Batch Analysis (V1)
```http
POST /api/v1/analyze/batch
```

**Request:**
```json
{
  "texts": [
    "This is amazing!",
    "Yeh bahut accha hai",
    "This is terrible"
  ]
}
```

**Response:**
```json
{
  "count": 3,
  "results": [
    {
      "original_text": "This is amazing!",
      "cleaned_text": "this is amazing",
      ...
    },
    ...
  ]
}
```

### Request Validation

- **Text length:** 1-5000 characters
- **Batch size:** 1-100 texts
- **Required fields:** All fields marked as required must be provided
- **Empty text:** Not allowed (validation error)

### Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid input data"
}
```

**422 Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Processing failed: [error message]"
}
```

## 💡 Usage Examples

### Example 1: Analyze Social Media Post (V2)
```python
import requests

post = "Aaj ka match dekha? Virat ne mara 6! What a shot! 🏏"

response = requests.post(
    "http://localhost:8000/api/v2/analyze",
    json={"text": post}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Route: {result['route']}")
print(f"Model: {result['model_used']}")
print(f"Hinglish: {result['language_detection']['is_hinglish']}")
```

### Example 2: Batch Processing Reviews (V2)
```python
reviews = [
    "Yeh product bahut accha hai!",
    "Delivery was terrible and late",
    "Amazing quality, highly recommend",
    "C'est magnifique! J'adore ça!"
]

response = requests.post(
    "http://localhost:8000/api/v2/analyze/batch",
    json={"texts": reviews}
)

for i, result in enumerate(response.json()['results']):
    print(f"Review {i+1}: {result['sentiment']} ({result['route']})")
```

### Example 3: Multilingual Support (V2)
```python
# Test multiple languages
texts = [
    "This is amazing!",
    "¡Esto es increíble!",
    "C'est magnifique!",
    "यह बहुत अच्छा है!"
]

for text in texts:
    response = requests.post(
        "http://localhost:8000/api/v2/analyze",
        json={"text": text}
    )
    result = response.json()
    print(f"Text: {text}")
    print(f"Language: {result['language_detection']['detected_language']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Model: {result['model_used']}")
    print("---")
```

## 📁 Project Structure

```
Code-mixed-NLP/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application
│   ├── schemas.py                       # Pydantic models
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                   # V1 text preprocessing
│   │   └── hybrid_preprocessor.py       # V2 spaCy + NLTK preprocessing
│   │
│   ├── language_detection/
│   │   ├── __init__.py
│   │   ├── detector.py                  # V1 rule-based detection
│   │   ├── fasttext_detector.py         # V2 FastText (176 languages)
│   │   ├── hingbert_detector.py         # Token-level Hinglish
│   │   └── lid.176.ftz                  # FastText model
│   │
│   ├── sentiment_analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py                  # V1 DistilBERT analyzer
│   │   ├── cmbert_analyzer.py           # V2 CM-BERT (Hinglish)
│   │   └── xlm_roberta_analyzer.py      # V2 XLM-RoBERTa (multilingual)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── hybrid_nlp_pipeline.py       # V2 smart routing pipeline
│   │
│   └── tests/
│       ├── conftest.py                  # Pytest configuration
│       ├── run_tests.py                 # Test runner
│       ├── test_api_integration.py      # 21 API tests
│       ├── test_core.py                 # Core component tests
│       ├── test_language_detection.py   # Language detection tests
│       ├── test_preprocessing.py        # Preprocessing tests
│       └── test_sentiment.py            # Sentiment tests
│
├── docs/                                # Documentation
│   ├── API_TEST_SAMPLES.md             # API examples
│   ├── DEPLOYMENT.md                   # Deployment guide
│   ├── MULTILINGUAL_API_v2.md          # V2 API documentation
│   └── PROJECT_STRUCTURE.md            # Project structure
│
├── scripts/                            # Utility scripts
│   ├── demo.py                         # Demo script
│   ├── setup.py                        # Setup script
│   ├── verify_setup.py                 # Environment verification
│   ├── start_server.bat                # Windows server launcher
│   └── test_deployment.ps1             # Deployment testing
│
├── models/                             # ML models directory
│   └── .gitkeep
│
├── pytest.ini                          # Pytest configuration
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version
├── Procfile                            # Deployment config
├── QUICKSTART.md                       # Quick start guide
└── README.md                           # This file
```

## 🧪 Testing

### Run Tests

**All Tests (21 API integration tests):**
```bash
pytest app/tests/ -v
```

**Specific Test File:**
```bash
pytest app/tests/test_api_integration.py -v
```

**Quick Test Summary:**
```bash
pytest app/tests/ -q
```

**With Coverage:**
```bash
pytest app/tests/ --cov=app --cov-report=html
```

### Test Configuration

- **pytest.ini** - Professional test configuration
- **Warning filters** - Suppresses external library warnings
- **Test markers** - Slow, integration, unit categories
- **Clean output** - 0 warnings, focused on actual results

### Test Coverage

✅ **21 API Integration Tests** (100% passing)
  - Health checks (V1 & V2)
  - V1 endpoints (preprocessing, language, sentiment, analyze, batch)
  - V2 endpoints (analyze, batch, languages, health)
  - Error handling and validation
  - Hinglish and multilingual text
  - All test categories covered

### Test Results

```
21 passed in ~30s
0 warnings
100% pass rate
```

## 🤖 Model Information

### V2 Models (Recommended)

#### 1. FastText Language Detection
- **Model:** Facebook's lid.176.ftz
- **Languages:** 176
- **Speed:** 10-20ms per text
- **Accuracy:** 95% (single language), 90% (code-mixed)
- **Size:** ~1 MB
- **Purpose:** Fast initial language routing

#### 2. HingBERT Token Detector
- **Model:** l3cube-pune/hindi-english-hing-bert
- **Accuracy:** 96% on code-mixed text
- **Speed:** 50-100ms per text
- **Size:** ~440 MB
- **Purpose:** Token-level Hinglish detection
- **Labels:** English, Hindi, Named Entity, Other

#### 3. CM-BERT Sentiment Analyzer
- **Model:** l3cube-pune/hing-sentiment-roberta
- **Languages:** Hinglish, Hindi, English
- **Accuracy:** 92% (Hinglish), 94% (English)
- **Speed:** 80-150ms per text
- **Size:** ~440 MB
- **Labels:** Positive, Negative, Neutral

#### 4. XLM-RoBERTa Sentiment Analyzer
- **Model:** cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
- **Languages:** 100+ (via transfer learning)
- **Accuracy:** 87% average
- **Speed:** 100-150ms per text
- **Size:** ~1.1 GB
- **Labels:** Positive, Negative, Neutral

### V1 Models (Legacy)

#### DistilBERT Sentiment
- **Model:** distilbert-base-uncased-finetuned-sst-2-english
- **Size:** 268 MB
- **Accuracy:** 93% on English text
- **Labels:** Positive, Negative

#### Rule-based Language Detection
- Hindi word dictionary (200+ words)
- English stopwords (NLTK)
- Devanagari script detection
- Named entity recognition patterns

### Smart Routing Strategy

```
Text Input
    ↓
FastText Detection (10-20ms)
    ↓
┌─────────────────────────────┐
│ Hinglish/Hindi/English?     │
├─────────────────────────────┤
│ YES → HingBERT + CM-BERT    │
│       • 92-96% accuracy      │
│       • 50-150ms total       │
│                              │
│ NO → XLM-RoBERTa            │
│      • 87% accuracy          │
│      • 100-150ms             │
└─────────────────────────────┘
```

## 🛠️ Development

### Adding New Features

1. **Create module** in appropriate directory
2. **Write unit tests** in `app/tests/`
3. **Add to pipeline** in `pipeline.py`
4. **Create API endpoint** in `main.py`
5. **Add integration tests** in `test_api_integration.py`

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Document functions with docstrings
- Write comprehensive tests

### Running in Development Mode

```bash
# Enable auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Testing with Swagger

1. Start the server
2. Open http://localhost:8000/docs
3. Click "Try it out" on any endpoint
4. Enter request data and click "Execute"
5. View response

## 📊 Performance

### V2 Performance Metrics

**First Request (Cold Start):**
- Model loading: 10-15 seconds
- Lazy loading: Models load on first use
- Subsequent requests: 200-350ms

**Processing Speed:**
- Single text (V2): 200-350ms
  - FastText detection: 10-20ms
  - HingBERT (if Hinglish): 50-100ms
  - Sentiment analysis: 80-150ms
- Batch (10 texts): 2-4 seconds
- Large text (1000 words): 400-600ms

**Resource Usage:**
- Memory: 3-4 GB (all models loaded)
- Memory (lazy): 1-2 GB (initial)
- CPU: Moderate (no GPU required)
- Disk: ~2.1 GB (model cache)

**Accuracy:**
- Hinglish sentiment: 92-94%
- English sentiment: 94%
- Multilingual sentiment: 87%
- Hinglish detection: 96%
- Language detection: 95%

### Railway FREE Tier Compatibility

✅ **Fits within Railway's FREE tier limits:**
- Memory: 8 GB limit (uses 3-4 GB)
- Image size: 4 GB limit (2.3 GB actual)
- Cold start: <30 seconds
- Idle timeout: Handles gracefully

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## 🌟 What's New in V2.0

### Major Improvements
- ✅ **37% accuracy improvement** for Hinglish (55% → 92%)
- ✅ **176 languages** supported (from 2)
- ✅ **Token-level detection** with HingBERT (96% accuracy)
- ✅ **Smart routing** for optimal model selection
- ✅ **Multilingual support** via XLM-RoBERTa
- ✅ **Professional testing** with pytest.ini and 0 warnings
- ✅ **Clean architecture** with lazy loading

### Backward Compatibility
- V1 endpoints still available and functional
- Gradual migration path provided
- No breaking changes for existing integrations

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

- **Yadnesh Teli** - Project development and implementation

## 🙏 Acknowledgments

### Models
- **L3Cube Pune** - HingBERT and CM-BERT models
- **Cardiff NLP** - XLM-RoBERTa multilingual sentiment
- **Facebook Research** - FastText language detection
- **Hugging Face** - Transformer model ecosystem

### Frameworks
- **FastAPI** - Modern web framework
- **spaCy** - Advanced NLP processing
- **PyTorch** - Deep learning framework
- **NLTK** - NLP utilities

### Platforms
- **Railway** - FREE tier deployment platform

## 📮 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Repository: YadneshTeli/Code-mixed-NLP

## 📚 Documentation

- **QUICKSTART.md** - Quick start guide
- **docs/DEPLOYMENT.md** - Deployment instructions
- **docs/MULTILINGUAL_API_v2.md** - Complete V2 API reference
- **docs/API_TEST_SAMPLES.md** - API usage examples
- **docs/PROJECT_STRUCTURE.md** - Project organization

---

**v2.0 - State-of-the-art multilingual NLP with specialized Hinglish support**  
**Made with ❤️ for the multilingual NLP community**
