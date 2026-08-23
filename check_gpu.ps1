# ─────────────────────────────────────────────────────────────
#  check_gpu.ps1 — run this FIRST, before writing any more code.
#  Windows PowerShell. No WSL, no Ubuntu needed.
#
#  Usage:   .\check_gpu.ps1
#  If PowerShell blocks it:
#           Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# ─────────────────────────────────────────────────────────────

$ok   = "$([char]0x2713)"
$bad  = "$([char]0x2717)"

function Say($msg, $colour = "White") { Write-Host $msg -ForegroundColor $colour }

Say "`n=== 1. NVIDIA driver and CUDA ===" Cyan
try {
    $smi = nvidia-smi
    $cuda = ($smi | Select-String "CUDA Version:\s*([\d.]+)").Matches.Groups[1].Value
    $drv  = ($smi | Select-String "Driver Version:\s*([\d.]+)").Matches.Groups[1].Value
    Say "  Driver : $drv"
    Say "  CUDA   : $cuda"
    if ([version]$cuda -ge [version]"12.8") {
        Say "  $ok CUDA 12.8+ — Blackwell (RTX 50-series) supported." Green
    } else {
        Say "  $bad CUDA $cuda is too old. RTX 5050 needs 12.8+ (driver R570+)." Red
        Say "     Update at nvidia.com/drivers, then re-run this script." Yellow
    }
} catch {
    Say "  $bad nvidia-smi not found. Install/update the NVIDIA driver." Red
}

Say "`n=== 2. Ollama service ===" Cyan
try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    Say "  $ok Ollama is running." Green
    Say "  Installed models:"
    foreach ($m in $tags.models) {
        $gb = [math]::Round($m.size / 1GB, 2)
        Say "    - $($m.name)  ($gb GB)"
    }
} catch {
    Say "  $bad Ollama not reachable on port 11434." Red
    Say "     Start it from the system tray, or run: ollama serve" Yellow
    exit 1
}

Say "`n=== 3. GPU offload check (THE ONE THAT MATTERS) ===" Cyan
Say "  Sending a warm-up request..." DarkGray
try {
    $body = @{ model = "qwen3:8b"; prompt = "Say OK."; stream = $false } | ConvertTo-Json
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" `
             -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300

    $ps = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" -TimeoutSec 10
    if (-not $ps.models) { Say "  $bad Nothing resident after warm-up." Red }
    foreach ($m in $ps.models) {
        $pct = if ($m.size -gt 0) { [math]::Round(100 * $m.size_vram / $m.size) } else { 0 }
        $vram = [math]::Round($m.size_vram / 1GB, 2)
        if ($pct -ge 99) {
            Say "  $ok $($m.name): $pct% GPU, $vram GB VRAM" Green
        } else {
            Say "  $bad $($m.name): only $pct% on GPU — spilling to CPU, ~5x slower" Red
            Say "     Try a smaller quant, or reduce num_ctx in models.yaml" Yellow
        }
    }
} catch {
    Say "  $bad Warm-up failed: $($_.Exception.Message)" Red
}

Say "`n=== 4. Throughput ===" Cyan
try {
    $body = @{
        model  = "qwen3:8b"
        prompt = "Explain in three sentences what a piping and instrumentation diagram is."
        stream = $false
    } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" `
         -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
    $rate = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1)
    Say "  $rate tokens/sec"
    if ($rate -ge 30) {
        Say "  $ok Healthy. Every remaining problem is a software problem." Green
    } elseif ($rate -ge 15) {
        Say "  ! Usable but slow. Check nothing else is using the GPU." Yellow
    } else {
        Say "  $bad Too slow — almost certainly running on CPU." Red
    }
} catch {
    Say "  $bad Benchmark failed: $($_.Exception.Message)" Red
}

Say "`n=== 5. Egress baseline ===" Cyan
$ext = Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
       Where-Object {
           $_.RemoteAddress -notmatch '^(127\.|::1|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|169\.254\.|fe80:)'
       }
Say "  External established connections right now: $($ext.Count)"
Say "  (This is your sovereignty baseline. During the demo it must read 0.)" DarkGray

Say "`nDone.`n" Cyan
