---
title: Code-Mixed NLP API
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# 🌐 Multilingual Hinglish NLP API

Production-ready REST API for processing code-mixed Hindi-English (Hinglish) text with multilingual support for 176 languages.

## 🎯 Features

- **92-96% accuracy** on Hinglish text
- **94% accuracy** on English text  
- **87% accuracy** on multilingual text
- **176 languages** supported via FastText
- **Specialized models**: HingBERT, CM-BERT, XLM-RoBERTa

## 🚀 Quick Start

### Test the API

```bash
# Health check
curl https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/health

# Analyze Hinglish text
curl -X POST https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/api/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Yaar this movie is too good! 🎬"}'

# Check supported languages
curl https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/api/v2/languages
```

### API Documentation

Visit the interactive docs: `https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/docs`

## 📋 Available Endpoints (14 Total)

### V1 Endpoints
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /api/v1/preprocess` - Text preprocessing
- `POST /api/v1/detect-language` - Language detection
- `POST /api/v1/analyze-sentiment` - Sentiment analysis
- `POST /api/v1/analyze` - Full Hinglish analysis
- `POST /api/v1/analyze/batch` - Batch processing

### V2 Endpoints (Multilingual)
- `GET /api/v2/health` - V2 health check
- `GET /api/v2/status` - Model status
- `GET /api/v2/warmup` - Pre-load models
- `POST /api/v2/preprocess` - V2 preprocessing
- `POST /api/v2/detect-language` - V2 language detection
- `POST /api/v2/analyze-sentiment` - V2 sentiment analysis
- `POST /api/v2/analyze` - Multilingual analysis
- `POST /api/v2/analyze/batch` - V2 batch processing
- `GET /api/v2/languages` - List 176 supported languages

## 🤖 Models Used

### Language Detection
- **FastText** - 176 language detection
- **HingBERT** - Token-level Hindi/English detection

### Sentiment Analysis
- **CM-BERT** - Hinglish sentiment (l3cube-pune/hing-sentiment-roberta)
- **XLM-RoBERTa** - Multilingual sentiment (cardiffnlp/twitter-xlm-roberta-base-sentiment)

## 📊 Example Usage

### Analyze Hinglish Text

**Request:**
```bash
curl -X POST /api/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Yaar this movie is too good! Bahut maza aaya! 🎬"
  }'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "scores": {
    "positive": 0.92,
    "neutral": 0.05,
    "negative": 0.03
  },
  "route": "HINGLISH",
  "model_used": "CM-BERT",
  "language_detection": {
    "detected_language": "hi",
    "confidence": 0.95,
    "is_hinglish": true
  }
}
```

### Detect Language

**Request:**
```bash
curl -X POST /api/v1/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'
```

**Response:**
```json
{
  "tokens": ["Bonjour", "comment", "allez-vous"],
  "labels": ["fr", "fr", "fr"],
  "dominant_language": "fr",
  "is_code_mixed": false,
  "statistics": {
    "language_distribution": {"fr": 100.0},
    "total_tokens": 3
  }
}
```

## 🎨 Use Cases

- **Social Media Analysis** - Analyze code-mixed tweets, posts
- **Customer Feedback** - Process multilingual reviews
- **Content Moderation** - Detect sentiment in mixed languages
- **Chatbot Enhancement** - Handle code-mixed queries
- **Research** - Study code-mixing patterns

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **Language**: Python 3.12
- **ML Libraries**: Transformers, spaCy, NLTK
- **Language Detection**: FastText, HingBERT
- **Sentiment Models**: CM-BERT, XLM-RoBERTa
- **Deployment**: Hugging Face Spaces (Docker)

## 📈 Performance

- **Inference Speed**: 100-500ms per request
- **Batch Processing**: Up to 100 texts per request
- **Model Size**: 1.1 GB (cached after first load)
- **Memory Usage**: ~2-3 GB with all models loaded

## 🔧 Configuration

Models are automatically downloaded and cached on first startup:
- FastText model (125 MB)
- spaCy model (12.8 MB)
- Transformer models (1.1 GB) - loaded on demand

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Developer

**Yadnesh Teli**
- GitHub: [@YadneshTeli](https://github.com/YadneshTeli)
- Repository: [Code-mixed-NLP](https://github.com/YadneshTeli/Code-mixed-NLP)

## 🤝 Contributing

Contributions are welcome! Please check the GitHub repository for contribution guidelines.

## 📚 Documentation

- [Full Documentation](https://github.com/YadneshTeli/Code-mixed-NLP/blob/v2-testing/README.md)
- [Deployment Guide](https://github.com/YadneshTeli/Code-mixed-NLP/blob/v2-testing/docs/DEPLOYMENT.md)
- [API Reference](https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/docs)

## 🙏 Acknowledgments

- HingBERT by l3cube-pune
- CM-BERT for Hinglish sentiment analysis
- XLM-RoBERTa by Cardiff NLP
- FastText by Facebook Research

---

**Status**: 🟢 Production Ready | **Version**: 2.0.0 | **API**: REST
