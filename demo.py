"""
Quick Demo Script - Hinglish NLP Pipeline
Run this to see the system in action!
"""

from app.pipeline import HinglishNLPPipeline

def demo():
    """Demonstrate the Hinglish NLP Pipeline capabilities"""
    
    print("\n" + "="*70)
    print("🌐 HINGLISH NLP PIPELINE - LIVE DEMO")
    print("="*70 + "\n")
    
    # Initialize pipeline
    print("📦 Initializing pipeline...")
    pipeline = HinglishNLPPipeline()
    
    # Demo samples
    samples = [
        {
            "text": "Yeh movie bahut accha hai! I loved it! 😊",
            "description": "Positive Code-Mixed Review"
        },
        {
            "text": "Aaj ka match kaisa tha? Virat ne century maari!",
            "description": "Hindi-Dominant Sports Commentary"
        },
        {
            "text": "This product is absolutely terrible and disappointing.",
            "description": "Negative English Review"
        },
        {
            "text": "Main bahut happy hoon because promotion mil gaya! 🎉",
            "description": "Mixed Personal Update"
        }
    ]
    
    # Process each sample
    for i, sample in enumerate(samples, 1):
        print(f"\n{'─'*70}")
        print(f"📝 Sample {i}: {sample['description']}")
        print(f"{'─'*70}\n")
        
        # Run analysis
        result = pipeline.process(sample['text'])
        
        # Display results
        print(f"📄 Original: {result['original_text']}")
        print(f"🧹 Cleaned:  {result['cleaned_text']}")
        print(f"📊 Tokens:   {result['token_count']} tokens")
        
        # Language info
        lang = result['language_detection']
        print(f"\n🌍 Language Analysis:")
        print(f"   • Dominant: {lang['dominant_language']}")
        print(f"   • Code-Mixed: {'Yes ✓' if lang['is_code_mixed'] else 'No ✗'}")
        
        # Show language distribution
        if lang['statistics']:
            print(f"   • Distribution:")
            for label, stats in lang['statistics'].items():
                label_name = {
                    'lang1': 'English',
                    'lang2': 'Hindi',
                    'ne': 'Named Entity',
                    'other': 'Other'
                }.get(label, label)
                bar = '█' * int(stats['percentage'] / 5)
                print(f"     - {label_name:12s}: {bar} {stats['percentage']:.1f}%")
        
        # Sentiment info
        sent = result['sentiment']
        sentiment_emoji = '😊' if sent['label'] == 'positive' else '😞' if sent['label'] == 'negative' else '😐'
        print(f"\n{sentiment_emoji} Sentiment: {sent['label'].upper()}")
        print(f"   • Confidence: {sent['confidence']*100:.2f}%")
        print(f"   • Scores:")
        for label, score in sent['scores'].items():
            bar = '█' * int(score * 20)
            print(f"     - {label.capitalize():8s}: {bar} {score*100:.1f}%")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📚 Key Features Demonstrated:")
    print("   ✓ Text preprocessing with emoji preservation")
    print("   ✓ Multi-language token detection")
    print("   ✓ Code-mixing identification")
    print("   ✓ High-confidence sentiment analysis")
    print("   ✓ Statistical language distribution")
    print("\n🚀 Ready for production use!")
    print("\n💡 Start the API server with: python app/main.py")
    print("   Then visit: http://localhost:8000/docs\n")


if __name__ == "__main__":
    demo()
