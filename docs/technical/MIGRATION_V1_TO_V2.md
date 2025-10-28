# 🔄 V1 to V2 Migration Plan

## 📋 Executive Summary

**Goal:** Migrate all V1 endpoints to V2 for 37% accuracy improvement and 176 language support

**Timeline:** Phased migration with comprehensive testing

**Risk:** Low - V1 endpoints remain for backward compatibility

---

## 📊 Current State Analysis

### V1 Endpoints (5 total - Legacy)
1. ❌ `POST /api/v1/preprocess` - Basic text cleaning
2. ❌ `POST /api/v1/detect-language` - Rule-based detection
3. ❌ `POST /api/v1/analyze-sentiment` - DistilBERT only
4. ❌ `POST /api/v1/analyze` - Full V1 pipeline
5. ❌ `POST /api/v1/analyze/batch` - V1 batch processing

### V2 Endpoints (4 total - Current)
1. ✅ `POST /api/v2/analyze` - Smart multilingual analysis
2. ✅ `POST /api/v2/analyze/batch` - Batch with smart routing
3. ✅ `GET /api/v2/languages` - Supported languages info
4. ✅ `GET /api/v2/health` - Component health check

### Missing V2 Equivalents
- No V2 equivalent for `/api/v1/preprocess` (standalone preprocessing)
- No V2 equivalent for `/api/v1/detect-language` (standalone detection)
- No V2 equivalent for `/api/v1/analyze-sentiment` (standalone sentiment)

---

## 🎯 Migration Strategy

### Phase 1: Create Missing V2 Endpoints ✅
Create V2 equivalents for all V1 standalone endpoints

**New V2 Endpoints to Create:**
1. `POST /api/v2/preprocess` - Hybrid preprocessing (spaCy + NLTK)
2. `POST /api/v2/detect-language` - FastText + HingBERT detection
3. `POST /api/v2/analyze-sentiment` - Smart sentiment routing

### Phase 2: Update V1 to Use V2 Pipeline ⏳
Refactor V1 endpoints to internally use V2 components

**Approach:**
- Keep V1 response structure (backward compatible)
- Use V2 models internally
- Map V2 responses to V1 format
- Maintain same endpoints

### Phase 3: Testing & Validation ⏳
Comprehensive testing of all endpoints

### Phase 4: Documentation Update ⏳
Update all documentation with migration guide

---

## 📝 Implementation Plan

### Task 1: Create V2 Preprocessing Endpoint
**File:** `app/main.py`

```python
@app.post("/api/v2/preprocess")
async def preprocess_v2(request: TextRequest):
    """
    V2 Preprocessing with hybrid spaCy + NLTK approach
    """
    # Use hybrid_preprocessor instead of cleaner
    # Return enhanced preprocessing info
```

**Response Schema:**
```json
{
  "original": "text",
  "processed": "cleaned text",
  "tokens": [...],
  "tokens_count": 10,
  "sentence_count": 2,
  "preprocessing_method": "hybrid"
}
```

### Task 2: Create V2 Language Detection Endpoint
**File:** `app/main.py`

```python
@app.post("/api/v2/detect-language")
async def detect_language_v2(request: TextRequest):
    """
    V2 Language Detection with FastText + HingBERT
    """
    # Use fasttext_detector and hingbert_detector
    # Return comprehensive language info
```

**Response Schema:**
```json
{
  "detected_language": "hi",
  "language_name": "Hindi",
  "confidence": 0.89,
  "is_hinglish": true,
  "is_reliable": true,
  "token_level_detection": {
    "tokens": [...],
    "labels": [...],
    "statistics": {...}
  }
}
```

### Task 3: Create V2 Sentiment Analysis Endpoint
**File:** `app/main.py`

```python
@app.post("/api/v2/analyze-sentiment")
async def analyze_sentiment_v2(request: TextRequest):
    """
    V2 Sentiment Analysis with smart model routing
    """
    # Use pipeline's smart routing
    # Return sentiment with model info
```

**Response Schema:**
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "confidence_level": "high",
  "scores": {...},
  "model_used": "CM-BERT",
  "route": "hinglish"
}
```

### Task 4: Update V1 Endpoints to Use V2 Components
**Files:** `app/main.py`

Update internal implementation while keeping V1 response format:
- `/api/v1/preprocess` → Use `hybrid_preprocessor`
- `/api/v1/detect-language` → Use `fasttext_detector` + `hingbert_detector`
- `/api/v1/analyze-sentiment` → Use pipeline routing
- `/api/v1/analyze` → Use full V2 pipeline
- `/api/v1/analyze/batch` → Use V2 batch processing

### Task 5: Create Response Mappers
**File:** `app/utils/response_mappers.py` (new)

```python
def map_v2_to_v1_preprocessing(v2_response):
    """Convert V2 preprocessing response to V1 format"""
    
def map_v2_to_v1_language(v2_response):
    """Convert V2 language detection to V1 format"""
    
def map_v2_to_v1_sentiment(v2_response):
    """Convert V2 sentiment to V1 format"""
```

---

## 🧪 Testing Strategy

### Test Suite Structure
```
app/tests/
├── test_v2_new_endpoints.py          # New V2 endpoints
├── test_v1_migration.py               # V1 using V2 internally
├── test_backward_compatibility.py     # V1 response format preserved
├── test_performance_comparison.py     # V1 vs V2 accuracy
└── test_integration_full.py           # End-to-end testing
```

### Test Scenarios

**1. V2 New Endpoints Testing**
- ✅ V2 preprocessing returns correct format
- ✅ V2 language detection uses FastText + HingBERT
- ✅ V2 sentiment uses smart routing
- ✅ All V2 endpoints handle errors properly
- ✅ Performance within acceptable limits

**2. V1 Migration Testing**
- ✅ V1 endpoints still work (backward compatible)
- ✅ V1 response format unchanged
- ✅ V1 endpoints use V2 models internally
- ✅ Improved accuracy on same test cases
- ✅ No breaking changes

**3. Accuracy Comparison**
- ✅ Hinglish sentiment: Old vs New
- ✅ Language detection: Old vs New
- ✅ Multilingual support improvements
- ✅ Edge cases handling

**4. Performance Testing**
- ✅ Response time acceptable
- ✅ Memory usage within limits
- ✅ Batch processing efficiency
- ✅ Model loading time

---

## 📈 Success Metrics

### Accuracy Improvements
- [ ] Hinglish sentiment: 55% → 92% (+37%)
- [ ] Language detection: Rule-based → 95%+ (FastText)
- [ ] Token-level detection: N/A → 96% (HingBERT)

### Coverage
- [ ] All V1 endpoints migrated
- [ ] All V2 equivalents created
- [ ] 100% test pass rate
- [ ] Zero breaking changes

### Performance
- [ ] Response time < 500ms (single request)
- [ ] Batch processing < 5s (10 texts)
- [ ] Memory usage < 4GB

---

## 🚀 Execution Checklist

### Preparation
- [ ] Review current V1 implementation
- [ ] Review V2 pipeline capabilities
- [ ] Create response mapper utilities
- [ ] Set up test data sets

### Implementation
- [ ] Create V2 preprocessing endpoint
- [ ] Create V2 language detection endpoint
- [ ] Create V2 sentiment endpoint
- [ ] Update V1 endpoints to use V2 internally
- [ ] Create response mappers

### Testing
- [ ] Write tests for new V2 endpoints
- [ ] Write V1 migration tests
- [ ] Run backward compatibility tests
- [ ] Performance benchmarking
- [ ] Accuracy comparison tests

### Documentation
- [ ] Update README.md
- [ ] Update API documentation
- [ ] Create migration guide
- [ ] Update examples

### Deployment
- [ ] Local testing complete
- [ ] All tests passing (100%)
- [ ] Documentation updated
- [ ] Ready for production

---

## 📌 Notes

**Backward Compatibility:**
- V1 endpoints will NOT be removed
- V1 response format will be maintained
- Existing integrations will continue to work
- V1 will internally use V2 models for better accuracy

**Migration Benefits:**
- 37% accuracy improvement on Hinglish
- 176 languages supported
- Smart model routing
- Better developer experience
- No breaking changes

**Timeline:**
- Phase 1: 2-3 hours (new endpoints)
- Phase 2: 2-3 hours (V1 migration)
- Phase 3: 2-4 hours (comprehensive testing)
- Phase 4: 1 hour (documentation)
- **Total:** ~8-10 hours

---

**Status:** Ready for execution
**Priority:** High
**Risk Level:** Low (backward compatible)
