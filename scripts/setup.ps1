# 创建并验证 distill conda 环境
# 用法: .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $ProjectRoot

Write-Host ">>> 创建 conda 环境 distill (Python 3.11) ..." -ForegroundColor Cyan
conda env create -f environment.yml --force 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "环境已存在，尝试更新 ..." -ForegroundColor Yellow
    conda env update -f environment.yml --prune
}

Write-Host ">>> 验证环境 ..." -ForegroundColor Cyan
conda run -n distill python scripts/verify_env.py

Write-Host ""
Write-Host "激活环境:" -ForegroundColor Green
Write-Host "  conda activate distill"
Write-Host ""
Write-Host "国内下载模型/数据集可设置镜像:" -ForegroundColor Green
Write-Host '  $env:HF_ENDPOINT = "https://hf-mirror.com"'
