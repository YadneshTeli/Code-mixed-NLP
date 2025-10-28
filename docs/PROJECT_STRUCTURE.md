# Project Structure

## Overview
This document describes the organized folder structure of the Code-mixed NLP project after cleanup and reorganization.

## Directory Structure

```
Code-mixed-NLP/
├── app/                          # Main application code
│   ├── language_detection/      # Language detection modules
│   ├── pipeline/                 # NLP pipeline implementation
│   ├── preprocessing/            # Text preprocessing
│   ├── sentiment_analysis/       # Sentiment analysis
│   └── main.py                   # FastAPI application
│
├── docs/                         # Documentation
│   ├── API_TEST_SAMPLES.md      # API testing examples
│   ├── DEPLOYMENT.md            # Deployment guide
│   ├── MULTILINGUAL_API_v2.md   # API documentation (v2.0)
│   └── PROJECT_STRUCTURE.md     # This file - project structure
│
├── models/                       # ML models and data
│   └── .gitkeep                 # Keep folder in git
│
├── scripts/                      # Utility scripts
│   ├── demo.py                  # Demo script for pipeline
│   ├── setup.py                 # Setup script
│   ├── verify_setup.py          # Environment verification
│   ├── start_server.bat         # Windows server launcher
│   └── test_deployment.ps1      # Deployment testing
│
├── app/tests/                    # Test suite (inside app folder)
│   ├── conftest.py              # Pytest configuration
│   ├── test_api_integration.py  # API integration tests (21 tests)
│   ├── test_core.py             # Core component tests
│   ├── test_language_detection.py # Language detection tests
│   ├── test_preprocessing.py    # Preprocessing tests
│   ├── test_sentiment.py        # Sentiment tests
│   └── run_tests.py             # Test runner
│
├── .env                         # Environment variables (not in repo)
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── Procfile                     # Deployment config
├── pytest.ini                   # Pytest configuration
├── QUICKSTART.md                # Quick start guide
├── README.md                    # Main readme
├── requirements.txt             # Python dependencies
└── runtime.txt                  # Python version for deployment
```

## Key Features

### Organized Structure
- **Separation of concerns**: Tests, docs, and scripts in dedicated folders
- **Clean root directory**: Only essential config files at root
- **Clear navigation**: Easy to find components by category

### Testing Infrastructure
- **Total: 21 tests** (100% passing)
  - API integration tests: 21 tests passing
- **Test configuration**: `pytest.ini` and `conftest.py` handle configuration and import paths
- **Easy execution**: Run all tests with `pytest app/tests/`
- **Zero warnings**: Clean test output with suppressed external library warnings

### Documentation
- All documentation consolidated in `docs/` folder
- API documentation, testing guides, and project summaries
- Deployment and setup instructions

### Scripts
- Utility scripts separated from application code
- Demo script updated to use new HybridNLPPipeline
- Setup and verification tools

## Recent Updates

### Documentation Cleanup
- Removed 8 outdated documentation files
- Kept only essential, current documentation
- PROJECT_STRUCTURE.md updated to reflect current state

### Test Configuration
- **pytest.ini** created for clean test output
- Warning filters configured for external libraries
- Test markers defined (slow, integration, unit)
- All 21 API integration tests passing with 0 warnings

### Cache Cleanup
- Removed all `__pycache__/` directories from app folder
- Clean codebase ready for production

## Running Tests

```bash
# All tests
pytest app/tests/ -v

# Specific test file
pytest app/tests/test_api_integration.py -v

# Quick test summary
pytest app/tests/ -q

# With coverage
pytest app/tests/ --cov=app --cov-report=html
```

## Running the Application

```bash
# Start the API server
uvicorn app.main:app --reload

# Run the demo
python scripts/demo.py

# Verify setup
python scripts/verify_setup.py
```

## Notes

- **Python version**: 3.12.6
- **Virtual environment**: Located in `venv/` (not tracked in git)
- **FastAPI**: Version 0.120.0
- **All tests passing**: 21/21 (100%) with 0 warnings
- **Test coverage**: Comprehensive API integration testing
- **Production ready**: Clean codebase with professional configuration
