# 🚀 Quick Start Guide - Hinglish NLP API

## Table of Contents
- [Starting the API](#starting-the-api)
- [Testing](#testing)
- [Running the Demo](#running-the-demo)
- [Using the API](#using-the-api)
- [File Structure](#file-structure)

---

## Starting the API

### Option 1: Using the Batch File (Easiest)
```bash
start_server.bat
```

### Option 2: Direct Python Command
```bash
python app/main.py
```

### Option 3: With Auto-Reload (Development)
```bash
uvicorn app.main:app --reload
```

**The server will start on:** `http://localhost:8000`

**Access Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Testing

### Run All Tests (93 total)
```bash
# Set PYTHONPATH and run all tests
$env:PYTHONPATH="."; pytest app/tests/ test_api_integration.py -v
```

### Run Unit Tests Only (72 tests)
```bash
$env:PYTHONPATH="."; pytest app/tests/ -v
```

### Run Integration Tests Only (21 tests)
```bash
pytest test_api_integration.py -v
```

### Run Specific Module Tests
```bash
# Preprocessing tests (19)
$env:PYTHONPATH="."; pytest app/tests/test_preprocessing.py -v

# Language detection tests (23)
$env:PYTHONPATH="."; pytest app/tests/test_language_detection.py -v

# Sentiment analysis tests (30)
$env:PYTHONPATH="."; pytest app/tests/test_sentiment_analysis.py -v
```

### Test with Coverage Report
```bash
$env:PYTHONPATH="."; pytest app/tests/ --cov=app --cov-report=html
```

---

## Running the Demo

### Live Demonstration
```bash
python demo.py
```

This will process 4 sample texts and show:
- Original and cleaned text
- Language detection results
- Sentiment analysis with confidence scores
- Statistical distribution

---

## Using the API

### Test with PowerShell

**1. Health Check:**
```powershell
curl http://localhost:8000/health
```

**2. Full Analysis:**
```powershell
$body = @{
    text = "Yeh movie bahut accha hai! I loved it!"
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/v1/analyze" `
     -H "Content-Type: application/json" `
     -Body $body
```

**3. Preprocessing Only:**
```powershell
$body = @{
    text = "Check out https://example.com! 😊 #amazing"
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/v1/preprocess" `
     -H "Content-Type: application/json" `
     -Body $body
```

**4. Language Detection Only:**
```powershell
$body = @{
    text = "Main bahut happy hoon today"
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/v1/detect-language" `
     -H "Content-Type: application/json" `
     -Body $body
```

**5. Sentiment Analysis Only:**
```powershell
$body = @{
    text = "This is absolutely amazing!"
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/v1/analyze-sentiment" `
     -H "Content-Type: application/json" `
     -Body $body
```

**6. Batch Processing:**
```powershell
$body = @{
    texts = @(
        "This is amazing!",
        "Yeh bahut accha hai",
        "This is terrible"
    )
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/v1/analyze/batch" `
     -H "Content-Type: application/json" `
     -Body $body
```

### Test with Python

```python
import requests

# Full analysis
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={"text": "Yeh movie bahut accha hai! I loved it!"}
)

print(response.json())
```

### Test with Browser
Simply go to http://localhost:8000/docs and use the "Try it out" button on any endpoint!

---

## File Structure

```
Code-mixed-NLP/
├── app/
│   ├── __init__.py
│   ├── main.py                          # ⭐ FastAPI application
│   ├── pipeline.py                      # ⭐ Integrated pipeline
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py                   # Text preprocessing
│   │
│   ├── language_detection/
│   │   ├── __init__.py
│   │   └── detector.py                  # Language detection
│   │
│   ├── sentiment_analysis/
│   │   ├── __init__.py
│   │   └── analyzer.py                  # Sentiment analysis
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_preprocessing.py        # 19 tests
│       ├── test_language_detection.py   # 23 tests
│       └── test_sentiment_analysis.py   # 30 tests
│
├── test_api_integration.py              # ⭐ 21 API integration tests
├── test_api.py                          # Manual API testing client
├── demo.py                              # ⭐ Live demonstration
├── start_server.bat                     # ⭐ Server startup script
├── requirements.txt                     # Python dependencies
├── README.md                            # ⭐ Complete documentation
├── PROJECT_SUMMARY.md                   # Technical summary
└── PROJECT_COMPLETION_CHECKLIST.md      # Progress tracker
```

**⭐ = Most important files**

---

## Common Tasks

### Check if API is Running
```bash
curl http://localhost:8000/health
```

### View All Available Endpoints
```bash
curl http://localhost:8000/
```

### Stop the Server
Press `Ctrl+C` in the terminal where the server is running

### Update Dependencies
```bash
pip install -r requirements.txt
```

### Activate Virtual Environment
```bash
# Windows PowerShell
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

---

## API Response Examples

### Health Check Response
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

### Full Analysis Response
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

---

## Troubleshooting

### Port Already in Use
If you see "Address already in use" error:

**Option 1: Use a different port**
```bash
uvicorn app.main:app --port 8001
```

**Option 2: Kill the process using port 8000**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Module Import Errors
Make sure you're in the project root directory and PYTHONPATH is set:
```bash
cd D:\Yadnesh-Teli\Projects\Code-mixed-NLP
$env:PYTHONPATH="."
```

### Model Not Found
The DistilBERT model (268MB) downloads on first run. Make sure you have:
- Internet connection
- ~500MB free disk space
- Wait ~60 seconds for first model load

---

## Performance Tips

- **First request takes ~60 seconds** (model loading) - subsequent requests are instant
- **Response time**: <300ms per request after warmup
- **Batch processing**: More efficient for multiple texts
- **Memory usage**: ~1.5GB with model loaded

---

## Next Steps

### Want to Deploy?

**Railway (Recommended)**
1. Create `Procfile`: `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
2. Push to GitHub
3. Connect repository at railway.app
4. Deploy!

**Render**
1. Same Procfile as above
2. Connect GitHub at render.com
3. Deploy

**Fly.io**
```bash
fly launch
fly deploy
```

### Want to Extend?

- Add translation module
- Implement caching (Redis)
- Add authentication
- Create web UI
- Fine-tune models on your data

---

## Support

- **Documentation**: See README.md for detailed docs
- **API Docs**: http://localhost:8000/docs (when server running)
- **Examples**: Run `python demo.py` for live examples

---

**Happy Coding! 🚀**
