# Prepare files for Hugging Face deployment
Write-Host "🏉 Preparing files for Hugging Face Spaces deployment..." -ForegroundColor Green

# Create deployment folder
$deployFolder = "huggingface-deployment"
if (Test-Path $deployFolder) {
    Remove-Item $deployFolder -Recurse -Force
}
New-Item -ItemType Directory -Path $deployFolder | Out-Null
Write-Host "✅ Created deployment folder" -ForegroundColor Green

# Copy main files
Copy-Item "app_hf.py" "$deployFolder\"
Copy-Item "Dockerfile" "$deployFolder\"
Copy-Item "requirements.txt" "$deployFolder\"
Copy-Item "README_HUGGINGFACE.md" "$deployFolder\README.md"
Write-Host "✅ Copied main files" -ForegroundColor Green

# Copy model
New-Item -ItemType Directory -Path "$deployFolder\src" -Force | Out-Null
Copy-Item "src\best.pt" "$deployFolder\src\"
Write-Host "✅ Copied model file" -ForegroundColor Green

# Copy frontend
New-Item -ItemType Directory -Path "$deployFolder\frontend" -Force | Out-Null
Copy-Item "frontend\index.html" "$deployFolder\frontend\"
Write-Host "✅ Copied frontend" -ForegroundColor Green

Write-Host "`n🎉 Files prepared successfully!" -ForegroundColor Green
Write-Host "`n📁 Files in '$deployFolder' folder:" -ForegroundColor Cyan
Get-ChildItem -Recurse $deployFolder | Select-Object FullName

Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Go to https://huggingface.co/new-space" -ForegroundColor White
Write-Host "2. Create a new Space with SDK = Docker" -ForegroundColor White
Write-Host "3. Upload all files from the '$deployFolder' folder" -ForegroundColor White
Write-Host "4. Wait for build to complete (~5-10 minutes)" -ForegroundColor White
Write-Host "5. Visit your Space URL to test!" -ForegroundColor White
Write-Host "`nFor detailed instructions, see: HUGGINGFACE_DEPLOY.md" -ForegroundColor Cyan
