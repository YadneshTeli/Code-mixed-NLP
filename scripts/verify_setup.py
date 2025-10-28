"""Quick verification test"""
import sys
from pathlib import Path

print("="*70)
print("🧪 Quick Verification Tests")
print("="*70)

# Test 1: HybridPreprocessor
print("\n1. Testing HybridPreprocessor...")
try:
    from app.preprocessing.hybrid_preprocessor import HybridPreprocessor
    print("   - Imported successfully")
    preprocessor = HybridPreprocessor()
    print("   - Instance created")
    result = preprocessor.preprocess("This is a test")
    print("   - Preprocessing complete")
    assert len(result['tokens']) > 0
    print(f"✅ HybridPreprocessor works! Tokens: {result['tokens']}")
except Exception as e:
    import traceback
    print(f"❌ HybridPreprocessor failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: FastTextDetector
print("\n2. Testing FastTextDetector...")
try:
    from app.language_detection.fasttext_detector import FastTextDetector
    detector = FastTextDetector()
    result = detector.detect("This is English text")
    assert result['language'] == 'en'
    print(f"✅ FastTextDetector works! Detected: {result['language']} ({result['confidence']:.2f})")
except Exception as e:
    print(f"❌ FastTextDetector failed: {e}")
    sys.exit(1)

# Test 3: Pipeline initialization
print("\n3. Testing HybridNLPPipeline...")
try:
    from app.pipeline.hybrid_nlp_pipeline import HybridNLPPipeline
    pipeline = HybridNLPPipeline()
    print("✅ HybridNLPPipeline initialized successfully!")
except Exception as e:
    print(f"❌ HybridNLPPipeline failed: {e}")
    sys.exit(1)

# Test 4: API imports
print("\n4. Testing API imports...")
try:
    from app.main import app
    print("✅ FastAPI app imports successfully!")
except Exception as e:
    print(f"❌ API import failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ All quick verification tests passed!")
print("="*70)
print("\nReady to run full test suite with pytest!")
