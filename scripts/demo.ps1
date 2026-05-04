<#
.SYNOPSIS
    End-to-end demo of the Helix SROP API.

.DESCRIPTION
    Exercises every feature of the running backend:
      1. Create a session
      2. Knowledge query (RAG + citations)
      3. Inspect the trace
      4. Account query
      5. Stateful follow-up
      6. Out-of-scope guardrail
      7. Idempotency replay
      8. Error cases (404 SESSION_NOT_FOUND, 404 TRACE_NOT_FOUND)

    The backend MUST already be running on $BaseUrl. Start it first with:
        python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

.PARAMETER BaseUrl
    Base URL of the running API. Defaults to http://127.0.0.1:8000.

.PARAMETER Timeout
    Per-request timeout in seconds. Defaults to 120.

.EXAMPLE
    .\scripts\demo.ps1

.EXAMPLE
    .\scripts\demo.ps1 -BaseUrl http://localhost:8000 -Timeout 180
#>
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$Timeout = 120
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8

function Section($title) {
    Write-Host ""
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host $title -ForegroundColor Cyan
    Write-Host "=========================================================" -ForegroundColor Cyan
}

function Show-Reply($r, $maxLen = 400) {
    if ($null -eq $r) { Write-Host "(null response)"; return }
    Write-Host "  routed_to : $($r.routed_to)" -ForegroundColor Yellow
    Write-Host "  trace_id  : $($r.trace_id)"
    $body = $r.reply
    if ($null -ne $body -and $body.Length -gt $maxLen) {
        $body = $body.Substring(0, $maxLen) + "..."
    }
    Write-Host "  reply     :"
    Write-Host "    $body"
}

# 0. Health check
Section "0. SANITY: /healthz"
try {
    $h = Invoke-RestMethod -Uri "$BaseUrl/healthz" -TimeoutSec 5
    Write-Host "  status: $($h.status)" -ForegroundColor Green
} catch {
    Write-Host "  Server is NOT reachable at $BaseUrl" -ForegroundColor Red
    Write-Host "  Start it first:" -ForegroundColor Red
    Write-Host "    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000" -ForegroundColor Red
    exit 1
}

# 1. Create session
Section "1. CREATE SESSION (user_id=u_demo, plan_tier=pro)"
$sess = Invoke-RestMethod -Uri "$BaseUrl/v1/sessions" -Method Post `
    -ContentType "application/json" -Body '{"user_id":"u_demo","plan_tier":"pro"}'
$SID = $sess.session_id
Write-Host "  session_id : $SID" -ForegroundColor Green

# 2. Turn 1 — knowledge query
Section "2. TURN 1 - Knowledge query (RAG + citations)"
Write-Host "  -> 'How do I rotate a deploy key?'"
try {
    $r1 = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"How do I rotate a deploy key?"}' -TimeoutSec $Timeout
    Show-Reply $r1
    $TID = $r1.trace_id
} catch {
    Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  BODY : $($_.ErrorDetails.Message)" -ForegroundColor Red
    $TID = $null
}

# 3. Inspect trace
if ($TID) {
    Section "3. INSPECT TRACE (debug-quality JSON)"
    $trace = Invoke-RestMethod -Uri "$BaseUrl/v1/traces/$TID"
    Write-Host "  routed_to            : $($trace.routed_to)" -ForegroundColor Yellow
    Write-Host "  retrieved_chunk_ids  : $($trace.retrieved_chunk_ids -join ', ')"
    Write-Host "  latency_ms           : $($trace.latency_ms)"
    Write-Host "  tool_calls           :"
    foreach ($tc in $trace.tool_calls) {
        Write-Host "    - $($tc.tool_name) | args: $($tc.args | ConvertTo-Json -Compress -Depth 4)"
    }
}

# 4. Turn 2 — account query
Section "4. TURN 2 - Account query (different sub-agent)"
Write-Host "  -> 'Show me my last 3 builds'"
try {
    $r2 = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"Show me my last 3 builds"}' -TimeoutSec $Timeout
    Show-Reply $r2
} catch {
    Write-Host "  ERROR: $($_.ErrorDetails.Message)" -ForegroundColor Red
}

# 5. Turn 3 — stateful follow-up
Section "5. TURN 3 - Stateful follow-up (no re-ask of plan_tier)"
Write-Host "  -> 'What plan am I on?'"
try {
    $r3 = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"What plan am I on?"}' -TimeoutSec $Timeout
    Show-Reply $r3
} catch {
    Write-Host "  ERROR: $($_.ErrorDetails.Message)" -ForegroundColor Red
}

# 6. Guardrail
Section "6. GUARDRAIL - Out-of-scope refusal (no LLM call)"
Write-Host "  -> 'Write me a poem about CI'"
try {
    $r4 = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"Write me a poem about CI"}' -TimeoutSec 30
    Show-Reply $r4
} catch {
    Write-Host "  ERROR: $($_.ErrorDetails.Message)" -ForegroundColor Red
}

# 7. Idempotency
Section "7. IDEMPOTENCY - Same key returns cached reply"
$headers = @{ "Idempotency-Key" = "demo-key-$(Get-Random)" }
Write-Host "  Idempotency-Key: $($headers.'Idempotency-Key')"
try {
    $i1 = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"How do I rotate a deploy key?"}' `
        -Headers $headers -TimeoutSec $Timeout
    $i2 = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/$SID" -Method Post `
        -ContentType "application/json" `
        -Body '{"content":"How do I rotate a deploy key?"}' `
        -Headers $headers -TimeoutSec $Timeout
    Write-Host "  first  trace_id : $($i1.trace_id)"
    Write-Host "  second trace_id : $($i2.trace_id)"
    if ($i1.trace_id -and $i1.trace_id -eq $i2.trace_id) {
        Write-Host "  MATCH (cached reply returned)" -ForegroundColor Green
    } else {
        Write-Host "  MISMATCH" -ForegroundColor Red
    }
} catch {
    Write-Host "  ERROR: $($_.ErrorDetails.Message)" -ForegroundColor Red
}

# 8. Error cases
Section "8. ERROR CASES"

Write-Host "  -- 404 SESSION_NOT_FOUND --"
try {
    Invoke-RestMethod -Uri "$BaseUrl/v1/chat/no-such-session" -Method Post `
        -ContentType "application/json" -Body '{"content":"hi"}'
    Write-Host "  unexpected: request succeeded" -ForegroundColor Red
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    Write-Host "  status: $code" -ForegroundColor Green
    Write-Host "  body  : $($_.ErrorDetails.Message)"
}

Write-Host ""
Write-Host "  -- 404 TRACE_NOT_FOUND --"
try {
    Invoke-RestMethod -Uri "$BaseUrl/v1/traces/no-such-trace"
    Write-Host "  unexpected: request succeeded" -ForegroundColor Red
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    Write-Host "  status: $code" -ForegroundColor Green
    Write-Host "  body  : $($_.ErrorDetails.Message)"
}

Section "DONE"
Write-Host "  All endpoints exercised against $BaseUrl"
Write-Host "  Session used in this run: $SID"
$SID | Out-File -Encoding utf8 .\demo_session_id.txt -Force
Write-Host "  (session_id saved to demo_session_id.txt for restart-survival demo)"
