"""
Quick Demo Script - Hinglish NLP Pipeline
Run this to see the system in action!
"""

from app.pipeline import HybridNLPPipeline

def demo():
    """Demonstrate the Hinglish NLP Pipeline capabilities"""
    
    print("\n" + "="*70)
    print("🌐 HINGLISH NLP PIPELINE - LIVE DEMO")
    print("="*70 + "\n")
    
    # Initialize pipeline
    print("📦 Initializing pipeline...")
    pipeline = HybridNLPPipeline()
    
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
        result = pipeline.analyze(sample['text'])
        
        # Display results
        print(f"📄 Original: {result['original_text']}")
        print(f"🧹 Cleaned:  {result['preprocessing']['cleaned_text']}")
        print(f"📊 Tokens:   {result['preprocessing']['tokens_count']} tokens")
        
        # Language info
        lang = result['language_detection']
        print(f"\n🌍 Language Analysis:")
        print(f"   • Language: {lang['language']}")
        print(f"   • Confidence: {lang['confidence']*100:.1f}%")
        print(f"   • Hinglish: {'Yes ✓' if lang.get('is_hinglish', False) else 'No ✗'}")
        
        # Sentiment info
        sent = result['sentiment_analysis']
        sentiment_emoji = '😊' if sent['sentiment'] == 'positive' else '😞' if sent['sentiment'] == 'negative' else '😐'
        print(f"\n{sentiment_emoji} Sentiment: {sent['sentiment'].upper()}")
        print(f"   • Confidence: {sent['confidence']*100:.2f}%")
        if 'probabilities' in sent:
            print(f"   • Scores:")
            for label, score in sent['probabilities'].items():
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
