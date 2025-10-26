# Hinglish NLP API - Test Sample Bodies

This document contains sample request bodies for testing all API endpoints.

## Base URL
- **Local**: `http://localhost:8000`
- **Railway**: `https://your-app.railway.app` (replace with your deployed URL)

---

## 1. Root Endpoint

**Endpoint**: `GET /`

**cURL**:
```bash
curl http://localhost:8000/
```

**Expected Response**:
```json
{
  "message": "Welcome to Hinglish NLP API",
  "version": "1.0.0",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "preprocess": "/preprocess",
    "detect_language": "/detect-language",
    "analyze_sentiment": "/analyze-sentiment",
    "analyze": "/analyze"
  }
}
```

---

## 2. Health Check

**Endpoint**: `GET /health`

**cURL**:
```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-10-27T12:00:00.000000"
}
```

---

## 3. Text Preprocessing

**Endpoint**: `POST /preprocess`

### Sample 1: Simple Hinglish Text
**Request Body**:
```json
{
  "text": "Aaj ka weather bohot acha hai yaar! 😊"
}
```

**cURL**:
```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Aaj ka weather bohot acha hai yaar! 😊\"}"
```

**Expected Response**:
```json
{
  "original_text": "Aaj ka weather bohot acha hai yaar! 😊",
  "cleaned_text": "aaj ka weather bohot acha hai yaar",
  "tokens": ["aaj", "ka", "weather", "bohot", "acha", "hai", "yaar"],
  "processing_time": 0.002
}
```

### Sample 2: Text with URLs and Mentions
**Request Body**:
```json
{
  "text": "Check out this link https://example.com and follow @username #hinglish"
}
```

### Sample 3: Text with Numbers and Special Characters
**Request Body**:
```json
{
  "text": "मैं 25 years old हूँ और मेरी salary ₹50000 है!!! 💰"
}
```

---

## 4. Language Detection

**Endpoint**: `POST /detect-language`

### Sample 1: Pure Hindi
**Request Body**:
```json
{
  "text": "मैं आज बाजार जा रहा हूँ।"
}
```

**cURL**:
```bash
curl -X POST http://localhost:8000/detect-language \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"मैं आज बाजार जा रहा हूँ।\"}"
```

**Expected Response**:
```json
{
  "text": "मैं आज बाजार जा रहा हूँ।",
  "language": "hindi",
  "confidence": 0.95,
  "is_code_mixed": false,
  "language_distribution": {
    "hindi": 100.0,
    "english": 0.0
  },
  "processing_time": 0.003
}
```

### Sample 2: Pure English
**Request Body**:
```json
{
  "text": "I am going to the market today."
}
```

### Sample 3: Hinglish (Code-Mixed)
**Request Body**:
```json
{
  "text": "Aaj main market जा रहा हूँ and shopping करूँगा।"
}
```

**Expected Response**:
```json
{
  "text": "Aaj main market जा रहा हूँ and shopping करूँगा।",
  "language": "hinglish",
  "confidence": 0.85,
  "is_code_mixed": true,
  "language_distribution": {
    "hindi": 60.0,
    "english": 40.0
  },
  "processing_time": 0.003
}
```

### Sample 4: Social Media Hinglish
**Request Body**:
```json
{
  "text": "Yaar ye movie toh ekdum mast thi! Must watch! 🎬"
}
```

---

## 5. Sentiment Analysis

**Endpoint**: `POST /analyze-sentiment`

### Sample 1: Positive Sentiment
**Request Body**:
```json
{
  "text": "This movie is absolutely amazing! Best film I've seen this year! 🌟"
}
```

**cURL**:
```bash
curl -X POST http://localhost:8000/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie is absolutely amazing! Best film I've seen this year! 🌟\"}"
```

**Expected Response**:
```json
{
  "text": "This movie is absolutely amazing! Best film I've seen this year! 🌟",
  "sentiment": "positive",
  "confidence": 0.9987,
  "scores": {
    "positive": 0.9987,
    "negative": 0.0013
  },
  "processing_time": 0.15
}
```

### Sample 2: Negative Sentiment
**Request Body**:
```json
{
  "text": "Yaar ye restaurant ki service bohot kharab hai. Waste of money! 😠"
}
```

**Expected Response**:
```json
{
  "text": "Yaar ye restaurant ki service bohot kharab hai. Waste of money! 😠",
  "sentiment": "negative",
  "confidence": 0.9856,
  "scores": {
    "positive": 0.0144,
    "negative": 0.9856
  },
  "processing_time": 0.15
}
```

### Sample 3: Neutral Sentiment
**Request Body**:
```json
{
  "text": "Main aaj office jaa raha hoon. Meeting hai 2 baje."
}
```

### Sample 4: Mixed Sentiment
**Request Body**:
```json
{
  "text": "Food was great but service was terrible. Mixed feelings about this place."
}
```

---

## 6. Full Analysis (Combined Pipeline)

**Endpoint**: `POST /analyze`

### Sample 1: Comprehensive Hinglish Text
**Request Body**:
```json
{
  "text": "Yaar aaj ka match ekdum zabardast tha! India ne Pakistan ko हरा दिया! 🏏🇮🇳"
}
```

**cURL**:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Yaar aaj ka match ekdum zabardast tha! India ne Pakistan ko हरा दिया! 🏏🇮🇳\"}"
```

**Expected Response**:
```json
{
  "original_text": "Yaar aaj ka match ekdum zabardast tha! India ne Pakistan ko हरा दिया! 🏏🇮🇳",
  "preprocessing": {
    "cleaned_text": "yaar aaj ka match ekdum zabardast tha india ne pakistan ko हरा दिया",
    "tokens": ["yaar", "aaj", "ka", "match", "ekdum", "zabardast", "tha", "india", "ne", "pakistan", "ko", "हरा", "दिया"],
    "token_count": 13
  },
  "language_detection": {
    "language": "hinglish",
    "confidence": 0.88,
    "is_code_mixed": true,
    "language_distribution": {
      "hindi": 55.0,
      "english": 45.0
    }
  },
  "sentiment_analysis": {
    "sentiment": "positive",
    "confidence": 0.9765,
    "scores": {
      "positive": 0.9765,
      "negative": 0.0235
    }
  },
  "metadata": {
    "total_processing_time": 0.18,
    "timestamp": "2025-10-27T12:00:00.000000"
  }
}
```

### Sample 2: Customer Review (Negative)
**Request Body**:
```json
{
  "text": "Delivery was very late aur product bhi damaged tha. Very disappointed with the service! 😤"
}
```

### Sample 3: Social Media Post (Positive)
**Request Body**:
```json
{
  "text": "Just watched the new movie! Climax ekdum mind-blowing tha yaar! 🔥 Highly recommended! 🎬✨"
}
```

### Sample 4: Product Review (Mixed)
**Request Body**:
```json
{
  "text": "Phone ka camera toh bahut acha hai but battery backup is really poor. Average product for the price."
}
```

---

## 7. Batch Analysis

**Endpoint**: `POST /batch-analyze`

### Sample: Multiple Texts
**Request Body**:
```json
{
  "texts": [
    "Yaar ye movie toh ekdum boring thi! Paisa waste! 😴",
    "Food was excellent! Service bhi acha tha. Will visit again! 😊",
    "Main aaj office nahi jaa raha. Sick leave le li hai.",
    "This product is amazing yaar! Best purchase ever! 💯"
  ]
}
```

**cURL**:
```bash
curl -X POST http://localhost:8000/batch-analyze \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Yaar ye movie toh ekdum boring thi! Paisa waste! 😴\", \"Food was excellent! Service bhi acha tha. Will visit again! 😊\"]}"
```

**Expected Response**:
```json
{
  "results": [
    {
      "text": "Yaar ye movie toh ekdum boring thi! Paisa waste! 😴",
      "language": "hinglish",
      "sentiment": "negative",
      "confidence": 0.9823
    },
    {
      "text": "Food was excellent! Service bhi acha tha. Will visit again! 😊",
      "language": "hinglish",
      "sentiment": "positive",
      "confidence": 0.9912
    },
    {
      "text": "Main aaj office nahi jaa raha. Sick leave le li hai.",
      "language": "hinglish",
      "sentiment": "neutral",
      "confidence": 0.7654
    },
    {
      "text": "This product is amazing yaar! Best purchase ever! 💯",
      "language": "hinglish",
      "sentiment": "positive",
      "confidence": 0.9956
    }
  ],
  "total_processed": 4,
  "total_time": 0.65
}
```

---

## Python Test Script

You can also use this Python script to test all endpoints:

```python
import requests
import json

BASE_URL = "http://localhost:8000"  # Change to your Railway URL

def test_endpoints():
    # Test samples
    samples = {
        "preprocess": {
            "text": "Aaj ka weather bohot acha hai yaar! 😊"
        },
        "detect-language": {
            "text": "Aaj main market जा रहा हूँ and shopping करूँगा।"
        },
        "analyze-sentiment": {
            "text": "This movie is absolutely amazing! Best film! 🌟"
        },
        "analyze": {
            "text": "Yaar aaj ka match ekdum zabardast tha! 🏏"
        }
    }
    
    for endpoint, data in samples.items():
        print(f"\n{'='*60}")
        print(f"Testing /{endpoint}")
        print(f"{'='*60}")
        
        response = requests.post(f"{BASE_URL}/{endpoint}", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    # Test health first
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    
    # Test all endpoints
    test_endpoints()
```

---

## PowerShell Test Script

```powershell
# Set base URL
$baseUrl = "http://localhost:8000"  # Change to your Railway URL

# Test health endpoint
Write-Host "`nTesting /health endpoint..." -ForegroundColor Cyan
$response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
$response | ConvertTo-Json

# Test preprocessing
Write-Host "`nTesting /preprocess endpoint..." -ForegroundColor Cyan
$body = @{
    text = "Aaj ka weather bohot acha hai yaar! 😊"
} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/preprocess" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json

# Test language detection
Write-Host "`nTesting /detect-language endpoint..." -ForegroundColor Cyan
$body = @{
    text = "Aaj main market जा रहा हूँ and shopping करूँगा।"
} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/detect-language" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json

# Test sentiment analysis
Write-Host "`nTesting /analyze-sentiment endpoint..." -ForegroundColor Cyan
$body = @{
    text = "This movie is absolutely amazing! Best film! 🌟"
} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/analyze-sentiment" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json

# Test full analysis
Write-Host "`nTesting /analyze endpoint..." -ForegroundColor Cyan
$body = @{
    text = "Yaar aaj ka match ekdum zabardast tha! 🏏"
} | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/analyze" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json
```

---

## Testing with Swagger UI

Once your server is running, visit:
- **Local**: http://localhost:8000/docs
- **Railway**: https://your-app.railway.app/docs

The Swagger UI provides an interactive interface where you can:
1. Click on any endpoint
2. Click "Try it out"
3. Enter the sample body
4. Click "Execute"
5. View the response

---

## Common Test Scenarios

### 1. E-commerce Review
```json
{
  "text": "Product quality toh acha hai but delivery was very slow. Overall okay experience."
}
```

### 2. Movie Review
```json
{
  "text": "Climax mind-blowing tha! Paisa vasool movie! Must watch in theaters! 🎬🔥"
}
```

### 3. Restaurant Review
```json
{
  "text": "Food was मस्त but prices are too high. Service bhi slow thi. Not worth it."
}
```

### 4. Social Media Comment
```json
{
  "text": "Bro this is epic! 😂 Can't stop laughing! Share karo sabke saath! 👏"
}
```

### 5. Customer Complaint
```json
{
  "text": "Customer service bohot kharab hai yaar! Phone pe koi reply nahi karta. Very frustrated! 😠"
}
```

---

## Notes

- **Processing Time**: Varies based on text length and model loading (first request may take 5-10 seconds)
- **Text Limits**: Maximum 1000 characters per request (configurable)
- **Batch Limits**: Maximum 100 texts per batch request
- **Rate Limiting**: Not implemented (add if needed for production)
- **Authentication**: Not required (add if needed for production)

---

## Troubleshooting

If you encounter errors:

1. **500 Internal Server Error**: Check server logs, model might not be loaded
2. **422 Validation Error**: Check request body format
3. **Connection Refused**: Ensure server is running
4. **Timeout**: First request loads model (~268MB), subsequent requests are fast

---

**Happy Testing! 🚀**
