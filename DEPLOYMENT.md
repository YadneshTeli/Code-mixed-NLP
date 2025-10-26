# Hinglish NLP API Deployment Guide

## 🚀 Quick Deploy Options

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

## ⚡ What Happens During Deployment

1. **Platform detects Python** from `runtime.txt`
2. **Installs dependencies** from `requirements.txt`
3. **Downloads DistilBERT model** (~268MB, first startup only)
4. **Starts server** using `Procfile` command
5. **Assigns public URL**

**First Deployment:** 3-5 minutes (model download)  
**Subsequent Deploys:** 1-2 minutes (cached)

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

### Issue 1: Torch Version Compatibility
**Error:** `ERROR: Could not find a version that satisfies the requirement torch==2.1.0`

**Cause:** Python 3.12.6 doesn't support torch 2.1.0 (requires 2.2.0+)

**Solution:** ✅ Already fixed! Requirements.txt uses `torch>=2.2.0` which is compatible with all platforms.

### Issue 2: Model Download Timeout
**Solution:** Some platforms have build timeouts. The 268MB model should download fine, but if it times out, consider using a smaller model or pre-downloading.

### Issue 3: Memory Limit
**Solution:** App needs ~1.5GB RAM. Most free tiers provide 512MB-1GB. Upgrade to paid tier if needed.

### Issue 4: Cold Starts
**Solution:** Free tiers sleep after inactivity. First request after sleep takes ~60 seconds (model reload). Paid tiers keep apps running.

---

## ✅ Deployment Success Indicators

When deployment succeeds, you should see:

1. ✅ Build logs show successful pip install
2. ✅ Model download completes (268MB)
3. ✅ Server starts with "Uvicorn running on..."
4. ✅ Health check returns `{"status": "healthy"}`
5. ✅ `/docs` shows Swagger UI
6. ✅ API responds to test requests

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
