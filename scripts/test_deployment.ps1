# Test Deployed API
# Usage: Update $baseUrl with your deployed URL and run this script

# Your deployed API URL (update after deployment)
$baseUrl = "http://localhost:8000"  # Change to your Railway/Render/Fly.io URL

Write-Host "`n🧪 Testing Deployed API: $baseUrl`n" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "1️⃣  Testing Health Check..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "   ✅ Health Check: " -NoNewline -ForegroundColor Green
    Write-Host $response.status
} catch {
    Write-Host "   ❌ Health Check Failed: $_" -ForegroundColor Red
}

# Test 2: Root Endpoint
Write-Host "`n2️⃣  Testing Root Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "   ✅ API Name: " -NoNewline -ForegroundColor Green
    Write-Host $response.name
} catch {
    Write-Host "   ❌ Root Endpoint Failed: $_" -ForegroundColor Red
}

# Test 3: Preprocessing
Write-Host "`n3️⃣  Testing Preprocessing..." -ForegroundColor Yellow
try {
    $body = @{
        text = "Check out https://example.com! 😊 #amazing"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/preprocess" `
                                   -Method Post `
                                   -Body $body `
                                   -ContentType "application/json"
    Write-Host "   ✅ Tokens: " -NoNewline -ForegroundColor Green
    Write-Host $response.token_count
} catch {
    Write-Host "   ❌ Preprocessing Failed: $_" -ForegroundColor Red
}

# Test 4: Language Detection
Write-Host "`n4️⃣  Testing Language Detection..." -ForegroundColor Yellow
try {
    $body = @{
        text = "Main bahut happy hoon today"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/detect-language" `
                                   -Method Post `
                                   -Body $body `
                                   -ContentType "application/json"
    Write-Host "   ✅ Dominant Language: " -NoNewline -ForegroundColor Green
    Write-Host $response.dominant_language
    Write-Host "   ✅ Code-Mixed: " -NoNewline -ForegroundColor Green
    Write-Host $response.is_code_mixed
} catch {
    Write-Host "   ❌ Language Detection Failed: $_" -ForegroundColor Red
}

# Test 5: Sentiment Analysis
Write-Host "`n5️⃣  Testing Sentiment Analysis..." -ForegroundColor Yellow
try {
    $body = @{
        text = "This is absolutely amazing!"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/analyze-sentiment" `
                                   -Method Post `
                                   -Body $body `
                                   -ContentType "application/json"
    Write-Host "   ✅ Sentiment: " -NoNewline -ForegroundColor Green
    Write-Host "$($response.label) ($([math]::Round($response.confidence * 100, 2))% confidence)"
} catch {
    Write-Host "   ❌ Sentiment Analysis Failed: $_" -ForegroundColor Red
}

# Test 6: Full Analysis
Write-Host "`n6️⃣  Testing Full Analysis..." -ForegroundColor Yellow
try {
    $body = @{
        text = "Yeh movie bahut accha hai! I loved it!"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/analyze" `
                                   -Method Post `
                                   -Body $body `
                                   -ContentType "application/json"
    Write-Host "   ✅ Tokens: " -NoNewline -ForegroundColor Green
    Write-Host $response.token_count
    Write-Host "   ✅ Dominant Language: " -NoNewline -ForegroundColor Green
    Write-Host $response.language_detection.dominant_language
    Write-Host "   ✅ Sentiment: " -NoNewline -ForegroundColor Green
    Write-Host "$($response.sentiment.label) ($([math]::Round($response.sentiment.confidence * 100, 2))%)"
} catch {
    Write-Host "   ❌ Full Analysis Failed: $_" -ForegroundColor Red
}

# Test 7: Batch Processing
Write-Host "`n7️⃣  Testing Batch Processing..." -ForegroundColor Yellow
try {
    $body = @{
        texts = @(
            "This is amazing!",
            "Yeh bahut accha hai",
            "This is terrible"
        )
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/analyze/batch" `
                                   -Method Post `
                                   -Body $body `
                                   -ContentType "application/json"
    Write-Host "   ✅ Processed: " -NoNewline -ForegroundColor Green
    Write-Host "$($response.count) texts"
} catch {
    Write-Host "   ❌ Batch Processing Failed: $_" -ForegroundColor Red
}

Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                  ✅ API TESTING COMPLETE                      ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`n📚 API Documentation: $baseUrl/docs" -ForegroundColor Cyan
Write-Host "📖 ReDoc: $baseUrl/redoc" -ForegroundColor Cyan
Write-Host ""
