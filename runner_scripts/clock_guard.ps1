# Clock stability guard for timed benchmarks - Windows port of clock_guard.sh:
# pin the GPU clocks, sample them at 1 Hz, and check each timed step against
# its own slice of the log.
#
# Dot-sourced, not executed:
#
#   Set-ClockTargets <dataset_key> [sm,mem]   resolve + validate target clocks
#   Lock-GpuClocks                            persistence mode + pin (needs admin)
#   Reset-GpuClocks                           restore the prior state
#   Start-ClockMonitor <csv>                  begin 1 Hz sampling
#   Stop-ClockMonitor                         end sampling
#   Get-ClockStamp [-End]                     window edge for Test-ClockSlice
#   Test-ClockSlice <from> <to> <label> <crit>  drift verdict for one time window
#   Show-ClockReport                          final per-step stability table
#
# Locking and resetting need an elevated (Administrator) console; sampling,
# checking and reporting do not.

$script:ClockSm = ''             # target SM/graphics clock, MHz ('' = not configured)
$script:ClockMem = ''            # target memory clock, MHz ('' = not locked/checked)
$script:ClockTolMhz = 15         # one clock step; anything beyond this is real drift
$script:ClockDriftPct = 1        # >this% of busy samples off target escalates to error
$script:ClockLocked = $false
$script:ClockPmRestore = ''      # persistence mode to put back on reset
$script:ClockCsv = ''
$script:ClockMonitor = $null
$script:ClockReportTsv = ''
if (-not $script:ClockConf) { $script:ClockConf = 'runner_scripts/gpu_clocks.conf' }

function Test-ClockAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Apply a setting; fail on refusal text despite exit 0, output kept in $script:ClockApplyOut.
function Invoke-ClockApply {
    param([string[]]$SmiArgs)
    $script:ClockApplyOut = (& nvidia-smi @SmiArgs 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { return $false }
    if ($script:ClockApplyOut -match 'not supported|Insufficient Permissions|Unable to') { return $false }
    return $true
}

function Test-ClockSupported {
    param([string]$Kind, [string]$Mhz)
    $offered = & nvidia-smi "--query-supported-clocks=$Kind" --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    foreach ($line in $offered) {
        if ($line.Trim() -eq $Mhz) { return $true }
    }
    return $false
}

# Resolve and validate target clocks; explicit "sm,mem" beats the per-GPU table.
function Set-ClockTargets {
    param([string]$DatasetKey, [string]$Explicit = '')
    $gpu = $DatasetKey -replace '^[^_]*_', ''
    $sm = ''
    $mem = ''

    if ($Explicit) {
        $parts = $Explicit -split ','
        $sm = $parts[0].Trim()
        if ($parts.Count -gt 1) { $mem = $parts[1].Trim() }
    }
    elseif (Test-Path $script:ClockConf) {
        foreach ($line in Get-Content $script:ClockConf) {
            $t = $line.Trim()
            if ($t -eq '' -or $t.StartsWith('#')) { continue }
            $cols = $t -split '\s+'
            if ($cols.Count -ge 2 -and $cols[0] -eq $gpu) {
                $sm = $cols[1]
                if ($cols.Count -ge 3) { $mem = $cols[2] }
                break
            }
        }
    }

    if (-not $sm) {
        Write-Warning "No clock target for '$gpu' in $($script:ClockConf) and none given."
        Write-Warning "Measure one: python runner_scripts/calibrate/calibrate_clocks.py"
        Write-Warning "or pass --lock-clocks SM[,MEM]. Continuing unlocked."
        return $false
    }
    if (-not (Test-ClockSupported 'gr' $sm)) {
        Write-Warning "SM clock $sm MHz is not a supported clock on this GPU."
        Write-Warning "List them: nvidia-smi --query-supported-clocks=gr --format=csv"
        return $false
    }
    if ($mem -and -not (Test-ClockSupported 'mem' $mem)) {
        Write-Warning "Memory clock $mem MHz is not a supported clock on this GPU."
        Write-Warning "List them: nvidia-smi --query-supported-clocks=mem --format=csv"
        return $false
    }

    $script:ClockSm = $sm
    $script:ClockMem = $mem
    return $true
}

# Pin the clocks; persistence mode first (not supported on the WDDM driver).
function Lock-GpuClocks {
    if (-not $script:ClockSm) { return $false }

    if (-not (Test-ClockAdmin)) {
        Write-Warning "Not an Administrator console - clocks stay unlocked."
        Write-Warning "Run these in an elevated shell first, then re-run:"
        Write-Warning "  nvidia-smi -lgc $($script:ClockSm),$($script:ClockSm)"
        if ($script:ClockMem) {
            Write-Warning "  nvidia-smi -lmc $($script:ClockMem),$($script:ClockMem)"
        }
        return $false
    }

    $script:ClockPmRestore = (& nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>$null | Select-Object -First 1)
    if (-not (Invoke-ClockApply @('-pm', '1'))) {
        Write-Warning "Could not enable persistence mode."
    }

    if (-not (Invoke-ClockApply @('-lgc', "$($script:ClockSm),$($script:ClockSm)"))) {
        Write-Warning "Failed to lock the SM clock to $($script:ClockSm) MHz."
        return $false
    }
    $script:ClockLocked = $true

    # Drop an unlockable memory clock from the drift check instead of failing.
    if ($script:ClockMem) {
        if (-not (Invoke-ClockApply @('-lmc', "$($script:ClockMem),$($script:ClockMem)"))) {
            if ($script:ClockApplyOut -match 'not supported') {
                Write-Warning "Locked memory clocks are not supported on this GPU; the memory clock stays on the driver default and is excluded from drift checks."
            }
            else {
                Write-Warning "Could not lock the memory clock to $($script:ClockMem) MHz; excluded from drift checks."
                Write-Warning "nvidia-smi said: $($script:ClockApplyOut.Trim())"
            }
            $script:ClockMem = ''
        }
    }

    $now = (& nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    $memNote = ''
    if ($script:ClockMem) { $memNote = ", MEM=$($script:ClockMem) MHz" }
    Write-Host "Clocks locked: SM=$($script:ClockSm) MHz$memNote  (now reading $now)"
    return $true
}

function Reset-GpuClocks {
    if (-not $script:ClockLocked) { return }
    $script:ClockLocked = $false
    if (Invoke-ClockApply @('-rgc')) {
        Write-Host "SM clock unlocked"
    } else {
        Write-Warning "Failed to reset the SM clock - run 'nvidia-smi -rgc' in an elevated shell"
    }
    Invoke-ClockApply @('-rmc') | Out-Null
    if ($script:ClockPmRestore -eq 'Disabled') {
        & nvidia-smi -pm 0 2>$null | Out-Null
    }
}

# One sampler process for the whole run.
function Start-ClockMonitor {
    param([string]$Csv)
    $script:ClockCsv = $Csv
    $script:ClockReportTsv = Join-Path (Split-Path -Parent $Csv) 'clock_stability.tsv'
    New-Item -ItemType File -Force -Path $script:ClockReportTsv | Out-Null
    $fields = 'timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_event_reasons.active'
    try {
        $script:ClockMonitor = Start-Process -FilePath 'nvidia-smi' `
            -ArgumentList @("--query-gpu=$fields", '--format=csv,nounits', '-lms', '1000') `
            -RedirectStandardOutput $Csv -RedirectStandardError "$Csv.err" `
            -NoNewWindow -PassThru
    } catch {
        $script:ClockMonitor = $null
    }
    Start-Sleep -Seconds 1
    if (-not $script:ClockMonitor -or $script:ClockMonitor.HasExited) {
        Write-Warning "Clock monitor died immediately; drift will not be checked."
        $script:ClockMonitor = $null
        return $false
    }
    return $true
}

function Stop-ClockMonitor {
    if ($script:ClockMonitor) {
        try {
            $script:ClockMonitor.Kill()
            $script:ClockMonitor.WaitForExit(5000) | Out-Null
        } catch { }
        $script:ClockMonitor = $null
    }
}

# Whole-second window edges for Test-ClockSlice.
function Get-ClockStamp {
    param([switch]$End)
    $now = Get-Date
    $base = $now.AddMilliseconds(-$now.Millisecond)
    if ($End) { return $base.AddMilliseconds(999) }
    return $base
}

# Window verdict for one step; returns $false only for critical DRIFT.
function Test-ClockSlice {
    param([datetime]$From, [datetime]$To, [string]$Label, [bool]$Critical = $true)
    if (-not $script:ClockCsv -or -not (Test-Path $script:ClockCsv)) { return $true }
    if (-not $script:ClockSm) { return $true }
    $sm = [int]$script:ClockSm
    $mem = 0
    if ($script:ClockMem) { $mem = [int]$script:ClockMem }
    $tol = [int]$script:ClockTolMhz

    $n = 0; $busy = 0; $drift = 0; $devworst = 0; $minsm = 0
    $memdrift = 0; $throttled = 0; $tmax = 0.0; $pmax = 0.0
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $styles = [System.Globalization.DateTimeStyles]::None

    foreach ($line in Get-Content $script:ClockCsv -ErrorAction SilentlyContinue) {
        $f = $line -split ','
        if ($f.Count -lt 7) { continue }
        $ts = [datetime]::MinValue
        if (-not [datetime]::TryParseExact($f[0].Trim(), 'yyyy/MM/dd HH:mm:ss.fff',
                                           $culture, $styles, [ref]$ts)) { continue }
        if ($ts -lt $From -or $ts -gt $To) { continue }
        $smNow = 0
        if (-not [int]::TryParse($f[1].Trim(), [ref]$smNow)) { continue }   # header / error text

        # Reason mask: 0x01 is GpuIdle; the 0xEC bits override a lock.
        $reasons = [uint64]0
        try { $reasons = [Convert]::ToUInt64(($f[6].Trim() -replace '^0[xX]', ''), 16) } catch { }
        $idle = ($reasons -band 0x1) -ne 0
        $bad = ($reasons -band 0xEC) -ne 0

        $n++
        $t = 0.0; $p = 0.0
        if ([double]::TryParse($f[3].Trim(), [ref]$t) -and $t -gt $tmax) { $tmax = $t }
        if ([double]::TryParse($f[4].Trim(), [ref]$p) -and $p -gt $pmax) { $pmax = $p }

        # Busy samples only.
        if ($idle) { continue }
        $busy++

        $dev = [math]::Abs($smNow - $sm)
        if ($dev -gt $devworst) { $devworst = $dev }
        if ($dev -gt $tol) { $drift++ }
        if ($minsm -eq 0 -or $smNow -lt $minsm) { $minsm = $smNow }

        $memNow = 0
        if ($mem -gt 0 -and [int]::TryParse($f[2].Trim(), [ref]$memNow)) {
            if ([math]::Abs($memNow - $mem) -gt $tol) { $memdrift++ }
        }
        if ($bad) { $throttled++ }
    }

    if ($n -eq 0) { return $true }

    # DRIFT needs >ClockDriftPct of busy samples off target, and at least 3.
    $pct = 0
    if ($busy -gt 0) { $pct = [int][math]::Ceiling($drift * 100.0 / $busy) }
    $verdict = 'OK'
    if ($throttled -gt 0 -or ($pct -gt $script:ClockDriftPct -and $drift -ge 3)) {
        $verdict = 'DRIFT'
    } elseif ($drift -gt 0 -or $memdrift -gt 0) {
        $verdict = 'BLIP'
    }

    $peak = '{0:0}C/{1:0.0}W' -f $tmax, $pmax
    Add-Content -Path $script:ClockReportTsv -Value (
        ($Label, $verdict, $n, $busy, $drift, $devworst, $minsm, $memdrift, $throttled, $peak) -join "`t")

    if ($verdict -eq 'BLIP') {
        Write-Warning "clock blip during ${Label}: $drift/$busy busy samples off $($sm)MHz (worst $($devworst)MHz, min $($minsm)MHz)"
    } elseif ($verdict -eq 'DRIFT') {
        $sev = 'WARNING'
        if ($Critical) { $sev = 'ERROR' }
        Write-Warning "${sev}: clocks drifted during $Label"
        Write-Warning "  $drift/$busy busy samples off $($sm)MHz ($pct%, worst $($devworst)MHz low, min $($minsm)MHz)"
        if ($memdrift -gt 0) {
            Write-Warning "  $memdrift samples off the $($mem)MHz memory clock"
        }
        if ($throttled -gt 0) {
            Write-Warning "  $throttled samples throttled by power/thermal limits (peak $peak)"
        }
        if ($Critical) {
            Write-Warning "  -> lower the lock and re-run this step"
        }
    }

    if ($verdict -eq 'DRIFT' -and $Critical) { return $false }
    return $true
}

# Final per-step table; returns $false when there is nothing to report.
function Show-ClockReport {
    if (-not $script:ClockReportTsv -or -not (Test-Path $script:ClockReportTsv)) { return $false }
    $rows = Get-Content $script:ClockReportTsv
    if (-not $rows) { return $false }
    Write-Host ''
    $target = 'unlocked'
    if ($script:ClockSm) { $target = "SM=$($script:ClockSm)" }
    if ($script:ClockMem) { $target += " MEM=$($script:ClockMem)" }
    Write-Host "CLOCK STABILITY  (target $target, tol $($script:ClockTolMhz)MHz)"
    Write-Host ('{0,-26} {1,-7} {2,8} {3,8} {4,9} {5,10}' -f 'STEP', 'STATUS', 'SAMPLES', 'OFF', 'WORST', 'PEAK')
    Write-Host ('{0,-26} {1,-7} {2,8} {3,8} {4,9} {5,10}' -f '----', '------', '-------', '---', '-----', '----')
    foreach ($row in $rows) {
        $c = $row -split "`t"
        if ($c.Count -lt 10) { continue }
        Write-Host ('{0,-26} {1,-7} {2,8} {3,8} {4,9} {5,10}' -f $c[0], $c[1], $c[3], $c[4], "$($c[5])MHz", $c[9])
    }
    return $true
}
