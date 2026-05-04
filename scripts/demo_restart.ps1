<#
.SYNOPSIS
    State-survival demo (the rubric-critical one).

.DESCRIPTION
    Step 1: run with -Phase before  -> creates a session, establishes
            plan_tier=enterprise in turn 1, writes session_id to
            demo_session_id.txt.
    Step 2: kill uvicorn (Ctrl+C) and restart it -- same command.
    Step 3: run with -Phase after   -> hits the SAME session_id and
            asks 'what plan am I on'. Reply must contain 'enterprise'
            even though we never re-told the agent the plan.

.EXAMPLE
    .\scripts\demo_restart.ps1 -Phase before
    # ... kill + restart uvicorn ...
    .\scripts\demo_restart.ps1 -Phase after
#>
param(
    [Parameter(Mandatory)]
    [ValidateSet("before","after")]
    [string]$Phase,

    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$Timeout = 120
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8

# Sanity
try {
    $h = Invoke-RestMethod "$BaseUrl/healthz" -TimeoutSec 5
} catch {
    Write-Host "Server not reachable at $BaseUrl. Start it first." -ForegroundColor Red
    exit 1
}

if ($Phase -eq "before") {
    Write-Host "=== BEFORE RESTART ===" -ForegroundColor Cyan
    $sess = Invoke-RestMethod -Uri "$BaseUrl/v1/sessions" -Method Post `
        -ContentType "application/json" `
        -Body '{"user_id":"u_persist","plan_tier":"enterprise"}'
    $SID = $sess.session_id
    Write-Host "session_id (saved to disk): $SID" -ForegroundColor Green

    # Establish state in turn 1
    $r = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"How do I rotate a deploy key?"}' -TimeoutSec $Timeout
    Write-Host "turn 1 routed_to: $($r.routed_to)"
    Write-Host "turn 1 reply (truncated): $($r.reply.Substring(0, [Math]::Min(200, $r.reply.Length)))..."

    $SID | Out-File -Encoding utf8 .\demo_session_id.txt -Force
    Write-Host ""
    Write-Host "Now: stop uvicorn (Ctrl+C in its terminal), restart it with the same command, then run:" -ForegroundColor Yellow
    Write-Host "    .\scripts\demo_restart.ps1 -Phase after" -ForegroundColor Yellow
}
else {
    Write-Host "=== AFTER RESTART ===" -ForegroundColor Cyan
    if (-not (Test-Path .\demo_session_id.txt)) {
        Write-Host "demo_session_id.txt not found. Run -Phase before first." -ForegroundColor Red
        exit 1
    }
    $SID = (Get-Content .\demo_session_id.txt -Raw).Trim()
    Write-Host "Reusing session_id: $SID" -ForegroundColor Green

    # Stateful follow-up: must answer 'enterprise' from persisted state
    $r = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"Remind me what plan I am on"}' -TimeoutSec $Timeout
    Write-Host ""
    Write-Host "routed_to: $($r.routed_to)" -ForegroundColor Yellow
    Write-Host "reply    : $($r.reply)"
    Write-Host ""
    if ($r.reply -match "(?i)enterprise") {
        Write-Host "PASS - state survived restart (reply contains 'enterprise')" -ForegroundColor Green
    } else {
        Write-Host "FAIL - reply does not mention the plan tier" -ForegroundColor Red
    }
}
