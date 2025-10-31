#!/bin/bash

# 🚀 Deploy to Hugging Face Spaces
# Run this script to quickly set up and deploy your Code-mixed NLP API

echo "🌟 Code-mixed NLP API - Hugging Face Deployment Script"
echo "========================================================"
echo ""

# Check if HF CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Hugging Face CLI not found!"
    echo "📦 Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Login to Hugging Face
echo "🔐 Logging into Hugging Face..."
echo "Please enter your Hugging Face token (get it from: https://huggingface.co/settings/tokens)"
huggingface-cli login

# Get username
echo ""
echo "📝 Please enter your Hugging Face username:"
read HF_USERNAME

# Space name
SPACE_NAME="code-mixed-nlp-api"
SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

echo ""
echo "🎯 Your Space will be created at: $SPACE_URL"
echo ""
echo "⚠️  NEXT STEPS:"
echo "1. Go to: https://huggingface.co/new-space"
echo "2. Fill in:"
echo "   - Space name: $SPACE_NAME"
echo "   - Space SDK: Docker"
echo "   - Space hardware: CPU basic (16 GB RAM, Free!)"
echo "   - Visibility: Public"
echo "3. Click 'Create Space'"
echo ""
echo "Press Enter when you've created the Space..."
read

# Clone the space
echo "📥 Cloning your Space..."
git clone https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME hf-space-temp
cd hf-space-temp

# Copy files (excluding git, cache, etc.)
echo "📦 Copying project files..."
rsync -av --progress ../ . \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='venv' \
  --exclude='env' \
  --exclude='.venv' \
  --exclude='hf-space-temp' \
  --exclude='node_modules' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  --exclude='.DS_Store' \
  --exclude='models/language_detection/*.bin'

# Copy the Space README (with metadata header)
cp ../README_SPACE.md README.md

# Update README with actual username
sed -i "s/YOUR_USERNAME/$HF_USERNAME/g" README.md

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
*.log
.DS_Store
models/language_detection/*.bin
EOF

# Git add and commit
echo "📝 Committing files..."
git add .
git commit -m "Initial deployment: Code-mixed NLP API v2.0"

# Push to Hugging Face
echo "🚀 Deploying to Hugging Face Spaces..."
git push

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "🎉 Your API will be live in 5-10 minutes at:"
echo "   $SPACE_URL"
echo ""
echo "📚 API Documentation:"
echo "   https://$HF_USERNAME-$SPACE_NAME.hf.space/docs"
echo ""
echo "🧪 Test endpoint:"
echo "   curl https://$HF_USERNAME-$SPACE_NAME.hf.space/health"
echo ""
echo "💡 Monitor build progress at: $SPACE_URL"
echo ""
echo "🎯 Once deployed, warmup models with:"
echo "   curl https://$HF_USERNAME-$SPACE_NAME.hf.space/api/v2/warmup"
echo ""

# Cleanup
cd ..
rm -rf hf-space-temp

echo "✨ Done! Check your Space for build progress."
