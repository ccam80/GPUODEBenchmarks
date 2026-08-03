# Windows MPGOS runner: enters the Visual Studio developer environment and calls nvcc.
param(
    # Exact aliases keep -a and -g unambiguous under prefix matching.
    [Alias('a')]
    [ValidateSet('performance', 'work-precision')]
    [string]$Analysis = 'performance',
    # A single value is a sweep ceiling (8, 32, ... <= n); a comma list runs exactly those Ns.
    [Alias('n')]
    [string]$Nmax = '16777216',
    [Alias('g')]
    [string]$Algorithm = 'all'
)

$ErrorActionPreference = 'Stop'

if ($Nmax -notmatch '^\d+(,\d+)*$') {
    Write-Host "-n/--nmax must be a positive integer or a comma list of them, got '$Nmax'"
    exit 1
}
if ($Nmax.Contains(',')) {
    $NValues = @($Nmax.Split(',') | ForEach-Object { [long]$_ })
} else {
    $NValues = @()
    $next = [long]8
    while ($next -le [long]$Nmax) {
        $NValues += $next
        $next = $next * 4
    }
}

# MPGOS solvers: RK4 (classical-rk4, fixed) and RKCK45 (cash-karp-54, adaptive).
$Solvers = switch ($Algorithm) {
    'all' { @('RK4', 'RKCK45') }
    'classical-rk4' { @('RK4') }
    'cash-karp-54' { @('RKCK45') }
    default {
        Write-Host "MPGOS does not support algorithm '$Algorithm'; skipping."
        exit 0
    }
}

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
$env:CUDA_MODULE_LOADING = 'EAGER'

Push-Location (Join-Path $PSScriptRoot '..\..')

function Enter-VsEnvironment {
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        return
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error "vswhere.exe not found; install Visual Studio with the C++ workload."
    }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        Write-Error "No Visual Studio installation with C++ tools found."
    }
    Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
}

# Mirrors GPU_ODE_MPGOS/Makefile.
function Build-Project {
    if (Test-Path "GPU_ODE_MPGOS\Lorenz.exe") {
        Remove-Item "GPU_ODE_MPGOS\Lorenz.exe" -Force
    }
    nvcc -o GPU_ODE_MPGOS\Lorenz.exe GPU_ODE_MPGOS\Lorenz.cu `
        -I"GPU_ODE_MPGOS\SourceCodes" `
        -O3 -std=c++17 --ptxas-options=-v --gpu-architecture=native `
        -lineinfo -maxrregcount=128
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nvcc build failed with exit code $LASTEXITCODE"
    }
}

# Solver and trajectory count are compile-time constants, so each point is a rebuild.
function Invoke-Point {
    param([string]$Solver, [long]$Nt, [switch]$Wp)
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"
    $content[14] = "#define SOLVER $Solver"
    $content[16] = "const int NT = $Nt;"
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"
    Build-Project
    if ($Wp) {
        & "GPU_ODE_MPGOS\Lorenz.exe" $Nt wp
    } else {
        & "GPU_ODE_MPGOS\Lorenz.exe" $Nt
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Lorenz.exe ($Solver) failed with exit code $LASTEXITCODE"
    }
}

Enter-VsEnvironment

if ($Analysis -eq 'work-precision') {
    foreach ($solver in $Solvers) {
        Invoke-Point -Solver $solver -Nt 32768 -Wp
    }
    Pop-Location
    return
}

foreach ($a in $NValues) {
    Write-Host "No. of trajectories = $a"
    foreach ($solver in $Solvers) {
        Invoke-Point -Solver $solver -Nt $a
    }
}

Pop-Location
