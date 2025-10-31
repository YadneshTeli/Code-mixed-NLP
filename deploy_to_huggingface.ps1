# 🚀 Deploy to Hugging Face Spaces (PowerShell)
# Run this script to quickly set up and deploy your Code-mixed NLP API

Write-Host "`n🌟 Code-mixed NLP API - Hugging Face Deployment Script" -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan

# Check if HF CLI is installed
$hfInstalled = Get-Command huggingface-cli -ErrorAction SilentlyContinue
if (-not $hfInstalled) {
    Write-Host "❌ Hugging Face CLI not found!" -ForegroundColor Red
    Write-Host "📦 Installing huggingface_hub..." -ForegroundColor Yellow
    pip install huggingface_hub
}

# Login to Hugging Face
Write-Host "🔐 Logging into Hugging Face..." -ForegroundColor Green
Write-Host "Please enter your Hugging Face token (get it from: https://huggingface.co/settings/tokens)`n" -ForegroundColor White
huggingface-cli login

# Get username
Write-Host "`n📝 Please enter your Hugging Face username:" -ForegroundColor Yellow
$HF_USERNAME = Read-Host

# Space name
$SPACE_NAME = "code-mixed-nlp-api"
$SPACE_URL = "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

Write-Host "`n🎯 Your Space will be created at: $SPACE_URL`n" -ForegroundColor Cyan
Write-Host "⚠️  NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Go to: https://huggingface.co/new-space" -ForegroundColor White
Write-Host "2. Fill in:" -ForegroundColor White
Write-Host "   - Space name: $SPACE_NAME" -ForegroundColor Gray
Write-Host "   - Space SDK: Docker" -ForegroundColor Gray
Write-Host "   - Space hardware: CPU basic (16 GB RAM, Free!)" -ForegroundColor Gray
Write-Host "   - Visibility: Public" -ForegroundColor Gray
Write-Host "3. Click 'Create Space'" -ForegroundColor White
Write-Host "`nPress Enter when you've created the Space..." -ForegroundColor Yellow
Read-Host

# Clone the space
Write-Host "📥 Cloning your Space..." -ForegroundColor Green
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" hf-space-temp
Set-Location hf-space-temp

# Copy files (excluding git, cache, etc.)
Write-Host "📦 Copying project files..." -ForegroundColor Green

$excludeDirs = @('.git', '__pycache__', 'venv', 'env', '.venv', 'hf-space-temp', 'node_modules', '.pytest_cache', '.DS_Store')
$excludePatterns = @('*.pyc', '*.egg-info')

Get-ChildItem -Path ..\ -Recurse | Where-Object {
    $item = $_
    $exclude = $false
    foreach ($dir in $excludeDirs) {
        if ($item.FullName -like "*\$dir\*" -or $item.Name -eq $dir) {
            $exclude = $true
            break
        }
    }
    if (-not $exclude) {
        foreach ($pattern in $excludePatterns) {
            if ($item.Name -like $pattern) {
                $exclude = $true
                break
            }
        }
    }
    -not $exclude
} | ForEach-Object {
    $relativePath = $_.FullName.Replace((Get-Location).Parent.FullName, "").TrimStart('\')
    $destination = Join-Path (Get-Location) $relativePath
    $destDir = Split-Path $destination -Parent
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    if ($_.PSIsContainer -eq $false) {
        Copy-Item $_.FullName $destination -Force
    }
}

# Copy the Space README (with metadata header)
Copy-Item ..\README_SPACE.md README.md -Force

# Update README with actual username
$readmeContent = Get-Content README.md -Raw
$readmeContent = $readmeContent -replace 'YOUR_USERNAME', $HF_USERNAME
Set-Content README.md $readmeContent

# Create .gitignore
@"
__pycache__/
*.py[cod]
*`$py.class
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
"@ | Set-Content .gitignore

# Git add and commit
Write-Host "📝 Committing files..." -ForegroundColor Green
git add .
git commit -m "Initial deployment: Code-mixed NLP API v2.0"

# Push to Hugging Face
Write-Host "🚀 Deploying to Hugging Face Spaces..." -ForegroundColor Green
git push

Write-Host "`n✅ Deployment initiated!`n" -ForegroundColor Green
Write-Host "🎉 Your API will be live in 5-10 minutes at:" -ForegroundColor Cyan
Write-Host "   $SPACE_URL`n" -ForegroundColor White
Write-Host "📚 API Documentation:" -ForegroundColor Cyan
Write-Host "   https://$HF_USERNAME-$SPACE_NAME.hf.space/docs`n" -ForegroundColor White
Write-Host "🧪 Test endpoint:" -ForegroundColor Cyan
Write-Host "   curl https://$HF_USERNAME-$SPACE_NAME.hf.space/health`n" -ForegroundColor White
Write-Host "💡 Monitor build progress at: $SPACE_URL`n" -ForegroundColor Yellow
Write-Host "🎯 Once deployed, warmup models with:" -ForegroundColor Yellow
Write-Host "   curl https://$HF_USERNAME-$SPACE_NAME.hf.space/api/v2/warmup`n" -ForegroundColor White

# Cleanup
Set-Location ..
Remove-Item hf-space-temp -Recurse -Force

Write-Host "✨ Done! Check your Space for build progress." -ForegroundColor Green
