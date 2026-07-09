param(
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = Join-Path $root "artifacts\example_runs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$results = @()
$examples = Get-ChildItem -Path (Join-Path $root "examples") -Filter "*.py" |
    Where-Object { $_.Name -match '^\d{2}_.*\.py$' } |
    Sort-Object Name

foreach ($ex in $examples) {
    $name = $ex.BaseName
    $logPath = Join-Path $logDir "$name.log"
    Write-Host "==> $name" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "uv"
    $psi.Arguments = "run python examples/$($ex.Name)"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = $root

    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        try { $proc.Kill($true) } catch {}
        $stdout = ""
        try { $stdout = $stdoutTask.Result } catch {}
        $stderr = ""
        try { $stderr = $stderrTask.Result } catch {}
        $exit = -1
        $status = "TIMEOUT"
    } else {
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        $exit = $proc.ExitCode
        $status = if ($exit -eq 0) { "PASS" } else { "FAIL" }
    }
    $sw.Stop()
    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)

    "=== STDOUT ===`n$stdout`n=== STDERR ===`n$stderr" | Set-Content -Path $logPath -Encoding UTF8

    $results += [pscustomobject]@{
        Example = $name
        Status  = $status
        Exit    = $exit
        Seconds = $elapsed
    }
    Write-Host "    [$status] exit=$exit  ${elapsed}s  log=$logPath"
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Yellow
$results | Format-Table -AutoSize
$results | Export-Csv -Path (Join-Path $logDir "_summary.csv") -NoTypeInformation
