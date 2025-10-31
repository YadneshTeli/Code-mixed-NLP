# 🚀 Deploying to Hugging Face Spaces

This guide will help you deploy your Code-mixed NLP API to Hugging Face Spaces with **16 GB RAM** (perfect for your transformer models!).

## ✨ Why Hugging Face Spaces?

- **16 GB RAM** - More than enough for 1.1 GB transformer models
- **Free forever** for public projects
- **Built for ML** - Optimized for transformers
- **Model caching** - Models stay loaded in memory
- **FastAPI support** - Your code works as-is
- **Auto-deployment** from git push

## 📋 Prerequisites

1. Create account at [huggingface.co](https://huggingface.co/join)
2. Install Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   ```

## 🔧 Step-by-Step Deployment

### Step 1: Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `code-mixed-nlp-api`
   - **License**: `mit` (or your choice)
   - **Space SDK**: Select **Docker**
   - **Space hardware**: Select **CPU basic** (16 GB RAM, Free!)
   - **Visibility**: Public (for free tier)

3. Click **Create Space**

### Step 2: Create Hugging Face Configuration File

The `README.md` in your Space root will serve as the Space card. We'll use the existing Dockerfile.

### Step 3: Push Your Code

```bash
# Login to Hugging Face
huggingface-cli login

# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/code-mixed-nlp-api
cd code-mixed-nlp-api

# Copy your project files (from your Code-mixed-NLP directory)
# Copy all files EXCEPT: .git, __pycache__, *.pyc, venv, etc.

# Add and commit
git add .
git commit -m "Initial deployment to Hugging Face Spaces"

# Push to Hugging Face
git push
```

### Step 4: Wait for Build

- Hugging Face will build your Docker image
- Build time: ~5-10 minutes (first time)
- Models will be cached after first load
- Your API will be live at: `https://YOUR_USERNAME-code-mixed-nlp-api.hf.space`

## 🎯 Hugging Face Specific Configuration

### Create `.spacesconfig.yaml` (Optional)

```yaml
sdk: docker
app_port: 8000
```

### Update Dockerfile (Already Compatible!)

Your existing `Dockerfile` works perfectly! Hugging Face will:
1. Build the Docker image
2. Download models during build
3. Start uvicorn on port 8000
4. Keep models loaded in memory (16 GB RAM!)

## 🧪 Testing Your Space

Once deployed, test endpoints:

```bash
# Health check
curl https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/health

# API docs
https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/docs

# Warmup models (no need - 16 GB RAM keeps them loaded!)
curl https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/api/v2/warmup

# V2 Analysis (will work perfectly with 16 GB RAM!)
curl -X POST https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/api/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Yaar this movie is too good!"}'
```

## 📊 Space Features

### Automatic Features:
- ✅ **API endpoints** - All 14 endpoints work
- ✅ **16 GB RAM** - No memory issues
- ✅ **Model caching** - Fast inference
- ✅ **HTTPS** - Secure by default
- ✅ **Logs** - View in Space settings
- ✅ **Metrics** - Request analytics
- ✅ **Custom domain** - Available (paid)

### Space UI (Optional):
You can add a Gradio interface for easier testing:
- Hugging Face provides built-in UI templates
- Users can test without curl commands
- Great for demos and portfolio

## 🔄 CI/CD with Hugging Face

### Auto-deployment from GitHub:

1. Add Hugging Face as remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/code-mixed-nlp-api
   ```

2. Push updates:
   ```bash
   git push hf v2-testing:main
   ```

3. Space rebuilds automatically!

## 💡 Tips for Hugging Face Spaces

### Optimize Build Time:
- Models download during build (cached)
- Use `.dockerignore` to exclude unnecessary files
- First build takes 10 minutes, subsequent builds ~2 minutes

### Environment Variables:
Set in Space Settings → Variables:
```bash
PYTHONUNBUFFERED=1
MODEL_CACHE_DIR=/data/models
```

### Persistent Storage:
- Use `/data` directory for model cache
- Survives container restarts
- Models load instantly after first run

### Monitoring:
- View logs: Space Settings → Logs
- Check metrics: Space Settings → Analytics
- Debug: Space Settings → Files

## 🆚 Hugging Face vs Railway

| Feature | Hugging Face | Railway |
|---------|-------------|---------|
| Free RAM | **16 GB** | ~1 GB |
| V2 Endpoints | ✅ Works | ❌ 502 errors |
| Model Loading | Instant | Timeout |
| Build Time | 5-10 min | 2-3 min |
| Community | ML-focused | General |
| Cost (Free) | Forever | Limited |

## 🎉 You're Done!

Your API is now running with **16 GB RAM** on Hugging Face Spaces:

- **Live URL**: `https://YOUR_USERNAME-code-mixed-nlp-api.hf.space`
- **API Docs**: `https://YOUR_USERNAME-code-mixed-nlp-api.hf.space/docs`
- **All 14 endpoints** working perfectly!
- **V2 transformer models** running smoothly!

## 📚 Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Docker SDK Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Spaces Examples](https://huggingface.co/spaces)

## 🤝 Need Help?

- Join [Hugging Face Discord](https://hf.co/join/discord)
- Check [Hugging Face Forum](https://discuss.huggingface.co)
- Visit [Spaces Documentation](https://huggingface.co/docs/hub/spaces)

---

**Note**: Make sure to update your repository URLs and documentation with the new Hugging Face Space URL once deployed!
