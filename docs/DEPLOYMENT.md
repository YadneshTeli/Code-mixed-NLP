# Multilingual Hinglish NLP API - Deployment Guide v2.0

## 🚀 Quick Deploy Options

### ⚠️ **IMPORTANT: V2.0 Changes**

**Version 2.0 introduces multilingual support with larger model sizes:**
- **Previous (v1.0):** 268 MB DistilBERT
- **Current (v2.0):** ~2.1 GB total (spaCy + FastText + HingBERT + CM-BERT + XLM-RoBERTa)
- **Image Size:** ~2.3 GB (under Railway's 4 GB FREE tier limit ✅)
- **RAM Required:** 3-4 GB (under Railway's 8 GB limit ✅)
- **First Load Time:** 10-15 seconds (models lazy load)

**Railway FREE tier is still compatible!** No upgrade needed.

### Option 1: Railway (Recommended - Easiest)

**Steps:**

1. **Install Railway CLI (Optional)**
   ```powershell
   npm install -g @railway/cli
   ```

2. **Deploy via GitHub (Easiest - No CLI needed)**
   - Go to https://railway.app
   - Click "Start a New Project"
   - Choose "Deploy from GitHub repo"
   - Select your `Code-mixed-NLP` repository
   - Railway will auto-detect the Procfile and deploy!

3. **Or Deploy via CLI**
   ```bash
   railway login
   railway init
   railway up
   ```

4. **Set Environment Variables (Optional)**
   - In Railway dashboard, go to Variables
   - Add: `PYTHON_VERSION=3.12.6`
   - Railway will auto-install from requirements.txt

5. **Get Your URL**
   - Railway will provide a URL like: `https://your-app.railway.app`
   - API docs will be at: `https://your-app.railway.app/docs`

**Expected Deploy Time:** 3-5 minutes

---

### Option 2: Render

**Steps:**

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name:** hinglish-nlp-api
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Click "Create Web Service"

**Free Tier:** Available (spins down after inactivity)

---

### Option 3: Fly.io

**Steps:**

1. **Install Fly CLI**
   ```powershell
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. **Login and Launch**
   ```bash
   fly auth login
   fly launch
   ```

3. **Follow prompts:**
   - App name: hinglish-nlp-api
   - Region: Choose closest to you
   - PostgreSQL: No
   - Redis: No

4. **Deploy**
   ```bash
   fly deploy
   ```

5. **Open app**
   ```bash
   fly open
   ```

---

### Option 4: Heroku

**Steps:**

1. **Install Heroku CLI**
   ```powershell
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create hinglish-nlp-api
   ```

3. **Deploy**
   ```bash
   git push heroku master
   ```

4. **Open app**
   ```bash
   heroku open
   ```

---

## 📋 Pre-Deployment Checklist

- ✅ `Procfile` created
- ✅ `runtime.txt` created
- ✅ `requirements.txt` up to date
- ✅ `.gitignore` configured (excludes .env, venv/, etc.)
- ✅ All tests passing (93/93)
- ✅ Code committed to Git

---

## 🔧 Environment Variables (Optional)

If deploying to production, you may want to set:

```bash
DEBUG=False
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=5000
```

**Note:** The app works fine with defaults, these are optional.

---

## ⚡ What Happens During Deployment (v2.0)

1. **Platform detects Python** from `runtime.txt`
2. **Installs dependencies** from `requirements.txt`
   - PyTorch CPU (200 MB)
   - spaCy + en_core_web_sm (12 MB)
   - FastText language model (126 MB)
3. **Downloads transformer models on first request** (lazy loading):
   - HingBERT (440 MB)
   - CM-BERT (440 MB)
   - XLM-RoBERTa (1.1 GB)
4. **Starts server** using `Procfile` command
5. **Assigns public URL**

**First Deployment:** 5-10 minutes (dependencies install)  
**First API Request:** 10-15 seconds (model download, then cached)  
**Subsequent Requests:** 200-350ms (models cached)  
**Subsequent Deploys:** 2-3 minutes (cached dependencies)

**Total Image Size:** ~2.3 GB ✅ (Railway FREE tier limit: 4 GB)

---

## 🧪 Testing Deployed API

Once deployed, test with:

```bash
# Replace with your actual URL
$url = "https://your-app.railway.app"

# Health check
curl "$url/health"

# Full analysis
$body = @{
    text = "Yeh movie bahut accha hai! I loved it!"
} | ConvertTo-Json

curl -X POST "$url/api/v1/analyze" `
     -H "Content-Type: application/json" `
     -Body $body
```

Or visit: `https://your-app.railway.app/docs` for Swagger UI

---

## 💰 Cost Estimates

| Platform | Free Tier | Paid (if needed) |
|----------|-----------|------------------|
| **Railway** | $5 credit/month | $0.000463/GB-hour |
| **Render** | 750 hours/month | $7/month |
| **Fly.io** | 3 shared-cpu VMs | $1.94/month per VM |
| **Heroku** | Discontinued free tier | $5-7/month |

**Recommendation:** Start with Railway's free tier ($5 credit)

---

## 🔍 Monitoring After Deployment

**Check logs:**

**Railway:**
```bash
railway logs
```

**Render:**
- View in dashboard under "Logs" tab

**Fly.io:**
```bash
fly logs
```

**Heroku:**
```bash
heroku logs --tail
```

---

## 🐛 Common Deployment Issues

### Issue 1: Image Size Exceeds Limit (CRITICAL)
**Error:** `Image of size 7.6 GB exceeded limit of 4.0 GB. Upgrade your plan to increase the image size limit.`

**Cause:** Full PyTorch with CUDA support is 2-3 GB, causing total image size to exceed Railway's 4 GB limit

**Solution:** ✅ Already fixed! Using CPU-only PyTorch (~200 MB instead of 2-3 GB)
- Added `--extra-index-url https://download.pytorch.org/whl/cpu` to requirements.txt
- This reduces total image size from 7.6 GB → ~2 GB (well under the 4 GB limit)
- **Note:** CPU inference is sufficient for this API - response times still under 1 second!

### Issue 2: Torch Version Compatibility
**Error:** `ERROR: Could not find a version that satisfies the requirement torch==2.1.0`

**Cause:** Python 3.12.6 doesn't support torch 2.1.0 (requires 2.2.0+)

**Solution:** ✅ Already fixed! Requirements.txt uses `torch>=2.2.0` which is compatible with all platforms.

### Issue 3: Pydantic Dependency Conflict
**Error:** `Cannot install pydantic==2.5.0 and pydantic-core==2.14.5 because these package versions have conflicting dependencies`

**Cause:** Pydantic 2.5.0 requires pydantic-core==2.14.1, not 2.14.5

**Solution:** ✅ Already fixed! Removed pinned pydantic-core version, using `pydantic>=2.5.0` to auto-resolve dependencies.

### Issue 4: Model Download Timeout (V2.0 Specific)

**Solution:** Models use lazy loading and are downloaded on first API request (not during build). The ~2 GB of models download in ~10-15 seconds. If your platform has request timeouts, the first request might fail - just retry after 30 seconds and models will be cached.

### Issue 5: Memory Limit (V2.0 Specific)

**Solution:** App needs ~3-4GB RAM with all models loaded. Railway FREE tier provides 8 GB, so you're covered! Other platforms may need paid tiers:
- Railway FREE: 8 GB ✅
- Render FREE: 512 MB ❌ (upgrade to $7/month for 2 GB, or $21/month for 4 GB)
- Fly.io FREE: 256 MB ❌ (upgrade to paid for 1+ GB)

### Issue 6: Cold Starts (V2.0 Specific)

**Solution:** 
- Free tiers sleep after 15-30 minutes of inactivity
- First request after sleep: ~30-60 seconds (loads ALL models)
- Subsequent requests: 200-350ms (models cached in memory)
- Paid tiers keep apps running 24/7 (no cold starts)

---

## ✅ Deployment Success Indicators (v2.0)

When deployment succeeds, you should see:

1. ✅ Build logs show successful pip install
2. ✅ Dependencies installed (~2.3 GB total)
3. ✅ Server starts with "Uvicorn running on..."
4. ✅ Health check returns `{"status": "healthy"}`
5. ✅ `/docs` shows Swagger UI with V2 endpoints
6. ✅ `/api/v2/languages` returns 176 supported languages
7. ✅ First API request takes 10-15 seconds (model download)
8. ✅ Subsequent requests take 200-350ms (models cached)

**Test V2 Endpoints:**
```bash
# Test new multilingual endpoint
curl "$url/api/v2/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Yaar this is awesome! Bahut accha hai!"}'

# Check supported languages
curl "$url/api/v2/languages"
```

---

## 🎯 Next Steps After Deployment

1. **Test all endpoints** via `/docs`
2. **Share API URL** with your team
3. **Monitor usage** in platform dashboard
4. **Set up custom domain** (optional)
5. **Enable HTTPS** (automatic on all platforms)
6. **Add monitoring** (optional: Sentry, LogRocket)

---

## 📞 Need Help?

**Platform-Specific Help:**
- Railway: https://docs.railway.app
- Render: https://render.com/docs
- Fly.io: https://fly.io/docs
- Heroku: https://devcenter.heroku.com

**Project Issues:**
- Check logs for errors
- Verify all tests pass locally
- Ensure requirements.txt is complete

---

**Ready to deploy? Choose your platform and follow the steps above!** 🚀
