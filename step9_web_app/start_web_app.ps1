$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到项目虚拟环境：$Python"
}

Set-Location -LiteralPath $PSScriptRoot
Write-Host "学生手册 RAG 正在启动：http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "首次提问需要加载本地模型，请耐心等待。按 Ctrl+C 停止。" -ForegroundColor DarkGray

$Server = Start-Process -FilePath $Python -ArgumentList ".\app.py" `
    -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow

try {
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        if ($Server.HasExited) {
            throw "本地服务启动失败，退出代码：$($Server.ExitCode)"
        }
        try {
            $Status = Invoke-RestMethod -Uri "http://127.0.0.1:7860/api/status" `
                -TimeoutSec 1
            if ($Status.ok) {
                $Ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $Ready) {
        throw "本地服务在预期时间内没有准备完成。"
    }

    Start-Process "http://127.0.0.1:7860"
    Wait-Process -Id $Server.Id
}
finally {
    if (-not $Server.HasExited) {
        Stop-Process -Id $Server.Id
    }
}
