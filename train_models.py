"""
Master Training Script for All Models
Trains HingBERT and CM-BERT models using LinCE dataset
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("🎓 CODE-MIXED NLP MODEL TRAINING SUITE")
print("=" * 80)

# Check dataset availability
DATA_DIR = Path("research files/archive")
if not DATA_DIR.exists():
    print(f"\n❌ ERROR: Dataset directory not found: {DATA_DIR}")
    print("Please ensure LinCE dataset is in 'research files/archive/'")
    sys.exit(1)

# Required dataset files
required_files = {
    "HingBERT LID": [
        "lid_hineng_train.csv",
        "lid_hineng_validation.csv", 
        "lid_hineng_test.csv"
    ],
    "CM-BERT Sentiment": [
        "sa_spaeng_train.csv",
        "sa_spaeng_validation.csv",
        "sa_spaeng_test.csv"
    ],
    "POS Tagging (Optional)": [
        "pos_hineng_train.csv",
        "pos_hineng_validation.csv",
        "pos_hineng_test.csv"
    ]
}

print("\n📁 Checking dataset files...")
missing_files = []
for task, files in required_files.items():
    print(f"\n{task}:")
    for file in files:
        file_path = DATA_DIR / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {file} (MISSING)")
            if task != "POS Tagging (Optional)":
                missing_files.append(file)

if missing_files:
    print(f"\n❌ ERROR: Missing required files: {missing_files}")
    sys.exit(1)

print("\n✅ All required dataset files found!")

# Training menu
print("\n" + "=" * 80)
print("TRAINING OPTIONS")
print("=" * 80)
print("1. Train HingBERT for Language Identification (Token-Level)")
print("   - Input: Hindi-English code-mixed text")
print("   - Output: Token-level language labels (lang1/lang2/ne/other)")
print("   - Dataset: ~1.5 MB training data")
print("   - Time: ~30-60 minutes on CPU, 5-10 minutes on GPU")
print()
print("2. Train CM-BERT for Sentiment Analysis")
print("   - Input: Code-mixed text")
print("   - Output: Sentiment (positive/negative/neutral)")
print("   - Dataset: ~3.1 MB training data (Spanish-English)")
print("   - Time: ~45-90 minutes on CPU, 10-15 minutes on GPU")
print()
print("3. Train Both Models (Recommended)")
print("4. Exit")
print("=" * 80)

choice = input("\nEnter your choice (1-4): ").strip()

if choice == "1":
    print("\n🚀 Starting HingBERT LID training...")
    import training.train_hingbert_lid as hingbert
    
elif choice == "2":
    print("\n🚀 Starting CM-BERT Sentiment training...")
    import training.train_cmbert_sentiment as cmbert
    
elif choice == "3":
    print("\n🚀 Starting training for both models...")
    print("\n" + "=" * 80)
    print("STEP 1/2: HingBERT Language Identification")
    print("=" * 80)
    import training.train_hingbert_lid as hingbert
    
    print("\n" + "=" * 80)
    print("STEP 2/2: CM-BERT Sentiment Analysis")
    print("=" * 80)
    import training.train_cmbert_sentiment as cmbert
    
    print("\n" + "=" * 80)
    print("✅ ALL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Models saved to:")
    print(f"  - HingBERT: ./models/trained/hingbert-lid-hinglish/")
    print(f"  - CM-BERT: ./models/trained/cmbert-sentiment-codemixed/")
    print(f"\n🎯 Next Steps:")
    print(f"  1. Update app/language_detection/hingbert_detector.py")
    print(f"  2. Update app/sentiment_analysis/cmbert_analyzer.py")
    print(f"  3. Test with: pytest app/tests/ -v")
    print(f"  4. Deploy to production")
    
elif choice == "4":
    print("\n👋 Exiting...")
    sys.exit(0)
    
else:
    print("\n❌ Invalid choice. Exiting...")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ TRAINING SESSION COMPLETE")
print("=" * 80)
