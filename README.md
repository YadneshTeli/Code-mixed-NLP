# 🌐 Hinglish NLP Pipeline & REST API

A comprehensive Natural Language Processing (NLP) pipeline for **Hinglish** (Hindi-English code-mixed) text analysis, featuring text preprocessing, language detection, sentiment analysis, and a production-ready REST API.

## ✨ Features

### 🔧 Text Preprocessing
- URL and mention removal
- Hashtag processing
- Emoji preservation
- Special character handling
- Smart tokenization

### 🌍 Language Detection
- Token-level language identification (Hindi/English)
- Named entity recognition
- Code-mixing detection
- Dominant language identification
- Statistical analysis of language distribution

### 😊 Sentiment Analysis
- Advanced transformer-based sentiment analysis (DistilBERT)
- Fallback rule-based sentiment detection
- Multi-label sentiment with confidence scores
- Support for Hinglish text
- Batch processing capability

### 🚀 REST API
- FastAPI-based REST API with auto-generated documentation
- 6 production-ready endpoints
- Request/response validation with Pydantic
- CORS support for web applications
- Comprehensive error handling
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
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

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

**Required Packages:**
- `fastapi>=0.104.1` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `pydantic>=2.5.0` - Data validation
- `transformers>=4.35.0` - NLP models
- `torch>=2.0.0` - PyTorch
- `pytest>=7.4.3` - Testing framework
- `requests>=2.31.0` - HTTP library

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

### 3. Test the API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Yeh movie bahut accha hai! I loved it!\"}"
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={"text": "Yeh movie bahut accha hai! I loved it!"}
)

print(response.json())
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "modules": {
    "preprocessing": true,
    "language_detection": true,
    "sentiment_analysis": true
  }
}
```

#### 2. Text Preprocessing
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

#### 3. Language Detection
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

#### 4. Sentiment Analysis
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

#### 5. Full Analysis
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

#### 6. Batch Analysis
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

### Example 1: Analyze Social Media Post
```python
import requests

post = "Aaj ka match dekha? Virat ne mara 6! What a shot! 🏏"

response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={"text": post}
)

result = response.json()
print(f"Sentiment: {result['sentiment']['label']}")
print(f"Languages: {result['language_detection']['dominant_language']}")
print(f"Code-mixed: {result['language_detection']['is_code_mixed']}")
```

### Example 2: Batch Processing Reviews
```python
reviews = [
    "Yeh product bahut accha hai!",
    "Delivery was terrible and late",
    "Amazing quality, highly recommend",
    "Waste of money, bilkul bakwas"
]

response = requests.post(
    "http://localhost:8000/api/v1/analyze/batch",
    json={"texts": reviews}
)

for i, result in enumerate(response.json()['results']):
    print(f"Review {i+1}: {result['sentiment']['label']}")
```

### Example 3: Language-Specific Processing
```python
text = "Main bahut khush hoon because I got promoted!"

# First detect language
lang_response = requests.post(
    "http://localhost:8000/api/v1/detect-language",
    json={"text": text}
)

lang_data = lang_response.json()
if lang_data['is_code_mixed']:
    print("Code-mixed text detected!")
    print(f"Dominant: {lang_data['dominant_language']}")
    
    # Then analyze sentiment
    sent_response = requests.post(
        "http://localhost:8000/api/v1/analyze-sentiment",
        json={"text": text}
    )
    print(f"Sentiment: {sent_response.json()['label']}")
```

## 📁 Project Structure

```
Code-mixed-NLP/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application
│   ├── pipeline.py                      # Integrated NLP pipeline
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py                   # Text preprocessing module
│   │
│   ├── language_detection/
│   │   ├── __init__.py
│   │   └── detector.py                  # Language detection module
│   │
│   ├── sentiment_analysis/
│   │   ├── __init__.py
│   │   └── analyzer.py                  # Sentiment analysis module
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_preprocessing.py        # 19 tests
│       ├── test_language_detection.py   # 23 tests
│       └── test_sentiment_analysis.py   # 30 tests
│
├── test_api_integration.py              # 21 API integration tests
├── test_api.py                          # API testing client
├── start_server.bat                     # Server startup script
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🧪 Testing

### Run All Tests (93 total)

**Unit Tests (72):**
```bash
$env:PYTHONPATH="."; pytest app/tests/ -v
```

**Integration Tests (21):**
```bash
pytest test_api_integration.py -v
```

**All Tests:**
```bash
$env:PYTHONPATH="."; pytest app/tests/ test_api_integration.py -v
```

### Test Coverage

- ✅ **72 Unit Tests**
  - 19 Preprocessing tests
  - 23 Language Detection tests
  - 30 Sentiment Analysis tests

- ✅ **21 Integration Tests**
  - Health & info endpoints
  - Preprocessing endpoint
  - Language detection endpoint
  - Sentiment analysis endpoint
  - Full analysis endpoint
  - Batch processing endpoint
  - Error handling

### Manual API Testing

Run the test client:
```bash
python test_api.py
```

This will test all endpoints and display results.

## 🤖 Model Information

### Sentiment Analysis Model

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

- **Architecture:** DistilBERT (Distilled BERT)
- **Size:** 268 MB
- **Training:** Fine-tuned on SST-2 sentiment dataset
- **Performance:** 
  - High accuracy on English text
  - Good performance on Hinglish text
  - 99%+ confidence on clear sentiments
- **Labels:** positive, negative

**Fallback:** Rule-based sentiment analysis using:
- Positive word list (excellent, amazing, wonderful, etc.)
- Negative word list (terrible, awful, horrible, etc.)

### Language Detection

**Approach:** Rule-based detection using:
- Hindi word dictionary (200+ common words)
- English stopwords (NLTK corpus)
- Devanagari script detection
- Named entity recognition patterns

**Features:**
- Token-level language labeling
- Code-mixing detection
- Dominant language identification
- Statistical distribution analysis

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

### Model Loading Time
- First request: ~60 seconds (DistilBERT model download & load)
- Subsequent requests: Instant (model cached in memory)

### Processing Speed
- Single text: ~100-300ms
- Batch (10 texts): ~1-2 seconds
- Large text (1000 words): ~500ms

### Resource Usage
- Memory: ~1.5 GB (with DistilBERT loaded)
- CPU: Moderate (no GPU required)
- Disk: ~300 MB (model cache)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

- Yadnesh Teli - Initial work

## 🙏 Acknowledgments

- Hugging Face for transformer models
- FastAPI for the excellent web framework
- NLTK for NLP utilities

## 📮 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Made with ❤️ for the Hinglish NLP community**

