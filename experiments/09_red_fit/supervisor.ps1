# 실행기가 죽거나 재부팅돼도 다시 돌린다. ALL_DONE이 생기면 멈춘다
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
New-Item -ItemType Directory -Force "$here\logs" | Out-Null
while (-not (Test-Path "$here\ALL_DONE")) {
    python -u runner.py *>> "$here\logs\supervisor.log"
    if (-not (Test-Path "$here\ALL_DONE")) { Start-Sleep -Seconds 60 }
}
