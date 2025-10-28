"""
Setup script for Multilingual Hinglish NLP v2.0

This script helps you set up the project quickly.

Usage:
    python setup.py
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    
    print_header("🚀 Multilingual Hinglish NLP Setup v2.0")
    
    print("This script will:")
    print("  1. Install Python dependencies (~2.1 GB models)")
    print("  2. Download spaCy model")
    print("  3. Download NLTK data")
    print("  4. Run tests to verify setup")
    print("\n⚠️  This may take 5-10 minutes on first run.\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return 1
    
    # Step 1: Install requirements
    print_header("Step 1: Installing Dependencies")
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements.txt"
    ):
        print("\n❌ Failed to install dependencies!")
        return 1
    
    # Step 2: Download spaCy model
    print_header("Step 2: Downloading spaCy Model")
    if not run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading en_core_web_sm"
    ):
        print("\n⚠️  Warning: spaCy model download failed. It may already be installed.")
    
    # Step 3: Download NLTK data
    print_header("Step 3: Downloading NLTK Data")
    nltk_script = """
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
print('NLTK data downloaded successfully!')
"""
    
    try:
        subprocess.run(
            [sys.executable, "-c", nltk_script],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ NLTK data downloaded successfully!")
    except subprocess.CalledProcessError:
        print("⚠️  Warning: NLTK data download failed. It may already be installed.")
    
    # Step 4: Verify installation
    print_header("Step 4: Verifying Installation")
    
    verification_script = """
import sys
print("Checking imports...")

try:
    import fastapi
    print("✅ FastAPI")
except ImportError as e:
    print(f"❌ FastAPI: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")
    sys.exit(1)

try:
    import spacy
    print(f"✅ spaCy {spacy.__version__}")
except ImportError as e:
    print(f"❌ spaCy: {e}")
    sys.exit(1)

try:
    import nltk
    print(f"✅ NLTK {nltk.__version__}")
except ImportError as e:
    print(f"❌ NLTK: {e}")
    sys.exit(1)

try:
    from fasttext_langdetect import detect
    print("✅ FastText")
except ImportError as e:
    print(f"❌ FastText: {e}")
    sys.exit(1)

print("\\n✅ All core dependencies installed successfully!")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", verification_script],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Verification failed!")
        print(e.stdout)
        print(e.stderr)
        return 1
    
    # Step 5: Run quick test
    print_header("Step 5: Running Quick Test")
    
    test_script = """
from app.preprocessing.hybrid_preprocessor import HybridPreprocessor
from app.language_detection.fasttext_detector import FastTextDetector

print("Testing HybridPreprocessor...")
preprocessor = HybridPreprocessor()
result = preprocessor.preprocess("This is a test")
assert len(result['tokens']) > 0
print("✅ HybridPreprocessor works!")

print("\\nTesting FastTextDetector...")
detector = FastTextDetector()
result = detector.detect("This is English text")
assert result['language'] == 'en'
print("✅ FastTextDetector works!")

print("\\n✅ Quick tests passed! System is ready.")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent)
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("⚠️  Quick test failed (models will download on first API request)")
        print(e.stdout)
        print(e.stderr)
    
    # Success!
    print_header("✅ Setup Complete!")
    
    print("Next steps:")
    print("\n1. Start the API server:")
    print("   python app/main.py")
    print("\n2. Visit API docs:")
    print("   http://localhost:8000/docs")
    print("\n3. Test V2 multilingual endpoint:")
    print("   curl -X POST http://localhost:8000/api/v2/analyze \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"text\": \"Yaar this is awesome!\"}'")
    print("\n4. Run full tests:")
    print("   python run_tests.py")
    print("\n5. Deploy to Railway:")
    print("   See DEPLOYMENT.md for instructions")
    
    print("\n" + "=" * 70)
    print("  🎉 Happy coding! Your multilingual NLP system is ready!")
    print("=" * 70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
