"""
Complete Model Setup - Download ALL Models
For local testing before Railway deployment
"""

import sys
import os
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_packages():
    """Verify required packages are installed"""
    print_header("🔍 Checking Required Packages")
    
    required = {
        'transformers': 'transformers',
        'torch': 'torch',
        'spacy': 'spacy',
        'fastapi': 'fastapi',
        'fasttext-wheel': 'fasttext'
    }
    
    missing = []
    for package, import_name in required.items():
        try:
            __import__(import_name.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages installed")
    return True

def cleanup_models():
    """Clean old models and incomplete downloads"""
    print_header("🧹 Cleanup Old Models & Duplicates")
    
    # Clean local models directory
    models_dir = Path("models")
    if models_dir.exists():
        import shutil
        shutil.rmtree(models_dir)
        print("✓ Removed old models directory")
    
    # Clean incomplete downloads from HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        incomplete_files = list(hf_cache.rglob("*.incomplete"))
        if incomplete_files:
            total_size = sum(f.stat().st_size for f in incomplete_files) / (1024**2)
            print(f"ℹ️  Found {len(incomplete_files)} incomplete downloads ({total_size:.1f} MB)")
            for f in incomplete_files:
                f.unlink()
            print(f"✓ Removed {len(incomplete_files)} incomplete download(s)")
        
        # Show cache size
        size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file()) / (1024**3)
        print(f"ℹ️  HuggingFace cache: {size:.2f} GB")
        
        # Only ask if cache is large
        if size > 2.0:
            response = input("  Remove HuggingFace cache? (y/n): ").strip().lower()
            if response == 'y':
                import shutil
                shutil.rmtree(hf_cache)
                print("✓ Removed HuggingFace cache")
            else:
                print("✓ Keeping existing cache (faster downloads)")
        else:
            print("✓ Cache size OK - keeping for faster downloads")
    
    return True

def download_model_1_fasttext():
    """Model 1: FastText Language Detection - 125 MB"""
    print_header("1️⃣  FastText Language Detection (125 MB)")
    
    print("Purpose: Detect language from 176 languages")
    print("Accuracy: 95%+")
    
    try:
        import urllib.request
        
        model_dir = Path("models/language_detection")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "lid.176.bin"
        
        if model_path.exists():
            size = model_path.stat().st_size / (1024**2)
            print(f"✅ Already exists ({size:.1f} MB)")
        else:
            print("📥 Downloading...")
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r   Progress: {percent:.1f}%", end='')
            
            urllib.request.urlretrieve(url, model_path, reporthook=progress)
            print()
            size = model_path.stat().st_size / (1024**2)
            print(f"✅ Downloaded ({size:.1f} MB)")
        
        # Verify
        from app.language_detection.fasttext_detector import FastTextDetector
        detector = FastTextDetector()
        result = detector.detect("Hello world")
        print(f"✓ Verified: Detected '{result['language']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_model_2_hingbert():
    """Model 2: HingBERT Token Detection - 440 MB"""
    print_header("2️⃣  HingBERT Token-Level Detection (440 MB)")
    
    print("Purpose: Token-level language tagging for Hinglish")
    print("Accuracy: 96%+ on code-mixed text")
    print("Model: l3cube-pune/hing-bert (SPECIFIC MODEL - NO ALTERNATIVES)")
    
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch
        
        model_name = "l3cube-pune/hing-bert"
        
        # Check if already cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
        if cache_dir.exists():
            print(f"\nℹ️  Model already in cache, loading...")
        else:
            print(f"\n📥 Downloading: {model_name}")
            print("   This may take 5-10 minutes...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Verify
        test_text = "Yaar ye movie toh mast thi"
        inputs = tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Downloaded ({params:.1f}M parameters)")
        print(f"✓ Verified: Output shape {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        print("   ⚠️  HingBERT is required - no fallback")
        return False

def download_model_3_hinglish_sentiment():
    """Model 3: CM-BERT Hinglish Sentiment - 500 MB"""
    print_header("3️⃣  CM-BERT Hinglish Sentiment Analysis (500 MB)")
    
    print("Purpose: Sentiment analysis for code-mixed Hinglish")
    print("Accuracy: 92%+ on Hinglish text")
    print("Model: l3cube-pune/hing-roberta (SPECIFIC MODEL - NO ALTERNATIVES)")
    
    try:
        from transformers import pipeline
        
        model_name = "l3cube-pune/hing-roberta"
        
        # Check if already cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
        if cache_dir.exists():
            print(f"\nℹ️  Model already in cache, loading...")
        else:
            print(f"\n📥 Downloading: {model_name}")
            print("   This may take 5-10 minutes...")
        
        sentiment = pipeline("sentiment-analysis", model=model_name, top_k=None)
        
        # Verify
        test_text = "Yaar ye toh mast hai!"
        result = sentiment(test_text)
        
        print(f"✅ Loaded: {model_name}")
        print(f"✓ Verified: {result[0] if isinstance(result[0], dict) else result[0][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        print("   ⚠️  CM-BERT is required - no fallback")
        return False

def download_model_4_xlm_roberta():
    """Model 4: XLM-RoBERTa Multilingual - 560 MB"""
    print_header("4️⃣  XLM-RoBERTa Multilingual Sentiment (560 MB)")
    
    print("Purpose: Multilingual sentiment analysis")
    print("Accuracy: 87% on multilingual text")
    print("Model: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual (SPECIFIC)")
    
    try:
        from transformers import pipeline
        
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
        
        # Check if already cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
        if cache_dir.exists():
            print(f"\nℹ️  Model already in cache, loading...")
        else:
            print(f"\n📥 Downloading: {model_name}")
            print("   This may take 5-10 minutes...")
        
        sentiment = pipeline("sentiment-analysis", model=model_name, top_k=None)
        
        # Verify
        test_texts = [
            "This is absolutely amazing!",
            "यह बहुत अच्छा है!",
        ]
        
        for text in test_texts:
            result = sentiment(text)
            print(f"✓ '{text[:30]}...' → {result[0][0]['label']}")
        
        print("✅ Downloaded and verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        print("   ⚠️  XLM-RoBERTa is required - no fallback")
        return False

def download_model_5_spacy():
    """Model 5: spaCy English - 13 MB"""
    print_header("5️⃣  spaCy English Model (13 MB)")
    
    print("Purpose: Text preprocessing and tokenization")
    print("Model: en_core_web_sm (SPECIFIC MODEL)")
    
    try:
        import spacy
        import subprocess
        
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ Already installed")
            doc = nlp("Test sentence")
            print(f"✓ Verified: {len(doc)} tokens")
            return True
            
        except OSError:
            print("\n📥 Installing en_core_web_sm...")
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                nlp = spacy.load("en_core_web_sm")
                print("✅ Installed")
                print(f"✓ Verified: Working")
                return True
            else:
                print(f"❌ Failed: {result.stderr}")
                print("   ⚠️  spaCy en_core_web_sm is required - no alternative")
                return False
                
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        print("   ⚠️  spaCy is required - no fallback")
        return False

def run_tests():
    """Run test suite to verify everything works"""
    print_header("🧪 Running Test Suite")
    
    try:
        import subprocess
        
        print("Running pytest...")
        print()
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "app/tests/", "-v", "--tb=short"],
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n⚠️  Some tests failed")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")
        return False

def show_summary():
    """Show deployment summary"""
    print_header("📊 Deployment Summary")
    
    sizes = {}
    total = 0
    
    # FastText
    fasttext = Path("models/language_detection/lid.176.bin")
    if fasttext.exists():
        size = fasttext.stat().st_size / (1024**2)
        sizes['FastText'] = size
        total += size
    
    # HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file()) / (1024**2)
        sizes['Transformer Models'] = size
        total += size
    
    # Display
    print(f"\n{'Model':<25} {'Size (MB)':>15}")
    print("-" * 42)
    for name, size in sizes.items():
        print(f"{name:<25} {size:>14.1f} MB")
    print("-" * 42)
    print(f"{'TOTAL':<25} {total:>14.1f} MB")
    print(f"{'Railway 4GB limit':<25} {4096:>14.1f} MB")
    print(f"{'Available':<25} {4096 - total:>14.1f} MB")
    
    if total < 2048:
        print(f"\n✅ Good size for Railway! ({total/1024:.2f} GB)")
    elif total < 4096:
        print(f"\n⚠️  Approaching limit ({total/1024:.2f} GB)")
    else:
        print(f"\n❌ Over Railway limit! ({total/1024:.2f} GB)")
    
    return total

def main():
    print("\n" + "🚀" * 35)
    print("  COMPLETE MODEL SETUP - SPECIFIC MODELS ONLY")
    print("  No Alternatives - Exact Models Required")
    print("🚀" * 35)
    
    print("\n📋 Specific Models to Download:")
    print("  1️⃣  FastText                        ~125 MB")
    print("      → lid.176.bin (176 languages)")
    print("  2️⃣  HingBERT                        ~440 MB")
    print("      → l3cube-pune/hing-bert")
    print("  3️⃣  CM-BERT (Hinglish Sentiment)    ~500 MB")
    print("      → l3cube-pune/hing-roberta")
    print("  4️⃣  XLM-RoBERTa                     ~560 MB")
    print("      → cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual")
    print("  5️⃣  spaCy                           ~13 MB")
    print("      → en_core_web_sm")
    print("  " + "─" * 66)
    print("  📦 TOTAL:                           ~1.6 GB")
    print("  ⏱️  Time:                            20-40 minutes")
    print("\n  ⚠️  NO FALLBACKS - All models are required")
    
    response = input("\n▶️  Download all models? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Setup cancelled")
        return
    
    # Check packages first
    if not check_packages():
        print("\n❌ Please install requirements first:")
        print("   pip install -r requirements.txt")
        return
    
    # Cleanup
    cleanup_models()
    
    # Download all models
    print("\n" + "⏳" * 35)
    print("  DOWNLOADING MODELS")
    print("⏳" * 35)
    
    results = {
        'FastText': download_model_1_fasttext(),
        'HingBERT': download_model_2_hingbert(),
        'Hinglish Sentiment': download_model_3_hinglish_sentiment(),
        'XLM-RoBERTa': download_model_4_xlm_roberta(),
        'spaCy': download_model_5_spacy()
    }
    
    # Summary
    print_header("✅ DOWNLOAD RESULTS")
    
    success = sum(1 for v in results.values() if v)
    total_models = len(results)
    
    print(f"\nSuccessfully downloaded: {success}/{total_models} models\n")
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    if success >= 3:  # At least core models
        # Show sizes
        total_size = show_summary()
        
        # Run tests
        print()
        response = input("▶️  Run tests to verify? (y/n): ").strip().lower()
        if response == 'y':
            tests_passed = run_tests()
        else:
            tests_passed = None
        
        print_header("🎯 NEXT STEPS")
        
        if tests_passed:
            print("\n✅ All models working and tests passing!")
        elif tests_passed is False:
            print("\n⚠️  Models downloaded but some tests failed")
        else:
            print("\n✅ Models downloaded successfully")
        
        print("\n1. Test API locally:")
        print("   uvicorn app.main:app --reload")
        print("   Open: http://localhost:8000/docs")
        
        print("\n2. For Railway deployment:")
        if total_size < 4096:
            print("   ✅ Current size OK for Railway")
        else:
            print("   ⚠️  May need to skip optional models")
            print("   Run: python setup_lean.py (for Railway)")
        
        print("\n3. Deploy:")
        print("   git add .")
        print("   git commit -m 'Add all models'")
        print("   git push")
        
    else:
        print("\n⚠️  Critical models failed to download")
        print("   Check your internet connection and try again")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
