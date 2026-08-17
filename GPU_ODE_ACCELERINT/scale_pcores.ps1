# Thread scaling, best of REPEATS, pinned to logical CPUs 0,2,..,14 (one per P-core).
param(
    [string]$Exe = "build\radau2a-mingw-ring_modulator-single.exe",
    [int]$N = 512,
    [int]$Repeats = 3,
    [string]$Schedule = "dynamic,1",
    [int[]]$Threads = @(1, 2, 3, 4, 5, 6, 7, 8)
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PATH = "$root\toolchain\openblas\bin;$root\toolchain\mingw64\bin;$env:PATH"
$env:OMP_SCHEDULE = $Schedule

function Invoke-Timed([int]$threads, [int64]$mask) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = Join-Path $root $Exe
    $psi.Arguments = "$threads $N"
    $psi.RedirectStandardOutput = $true
    $psi.UseShellExecute = $false
    $psi.EnvironmentVariables["OMP_SCHEDULE"] = $Schedule
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.ProcessorAffinity = [IntPtr]$mask
    $out = $p.StandardOutput.ReadToEnd()
    $p.WaitForExit()
    if ($out -notmatch '(?m)^# status: (\d+)/(\d+) converged') {
        throw "no status line for $threads threads"
    }
    $converged = [int]$Matches[1]
    if ($out -notmatch '(?m)^Time: (\S+)') { throw "no time line for $threads threads" }
    [pscustomobject]@{ Seconds = [double]$Matches[1]; Converged = $converged }
}

"threads`tbest_s`tspeedup`tconverged"
$base = 0.0
foreach ($t in $Threads) {
    $mask = 0L
    for ($i = 0; $i -lt $t; $i++) { $mask = $mask -bor ([int64]1 -shl (2 * $i)) }
    $runs = 1..$Repeats | ForEach-Object { Invoke-Timed $t $mask }
    $best = ($runs | Measure-Object -Property Seconds -Minimum).Minimum
    if ($t -eq $Threads[0]) { $base = $best }
    "{0}`t{1:N3}`t{2:N2}`t{3}/{4}" -f $t, $best, ($base / $best), $runs[0].Converged, $N
}
