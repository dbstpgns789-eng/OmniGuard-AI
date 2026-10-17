# 실행기가 죽거나 PC가 재시작돼도 다시 돌린다. 모든 작업이 끝나면(ALL_DONE) 멈춘다
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
while (-not (Test-Path "$here\ALL_DONE")) {
    python -u runner.py *>> "$here\logs\supervisor.log"
    if (-not (Test-Path "$here\ALL_DONE")) { Start-Sleep -Seconds 60 }
}
