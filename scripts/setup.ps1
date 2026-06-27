# 创建并验证 distill conda 环境
# 用法: .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = "$env:USERPROFILE\.conda\envs\distill\python.exe"
$Pip = "$env:USERPROFILE\.conda\envs\distill\Scripts\pip.exe"

Set-Location $ProjectRoot

Write-Host ">>> 创建/更新 conda 环境 distill ..." -ForegroundColor Cyan
conda env create -f environment.yml 2>$null
if ($LASTEXITCODE -ne 0) {
    conda env update -f environment.yml --prune
}

Write-Host ">>> 安装 PyTorch (cu128, ~2.7GB, 需要稳定网络) ..." -ForegroundColor Cyan
& $Pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch 下载失败，请重试: .\scripts\setup.ps1" -ForegroundColor Red
    exit 1
}

Write-Host ">>> 验证环境 ..." -ForegroundColor Cyan
& $Python scripts/verify_env.py

Write-Host ""
Write-Host "激活环境:" -ForegroundColor Green
Write-Host "  conda activate distill"
Write-Host ""
Write-Host "国内下载模型/数据集:" -ForegroundColor Green
Write-Host '  $env:HF_ENDPOINT = "https://hf-mirror.com"'
