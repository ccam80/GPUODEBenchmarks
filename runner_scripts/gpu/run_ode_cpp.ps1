# PowerShell script for running C++ (MPGOS) ODE benchmarks
#
# The MPGOS Makefile is GNU-make syntax, which nmake cannot parse, and nvcc
# on Windows needs MSVC's cl.exe as its host compiler. Instead of requiring
# make, this script enters the Visual Studio developer environment (located
# via vswhere) and invokes nvcc directly with the Makefile's flags.
param(
    [Parameter(Mandatory=$true)]
    [int]$MaxA,
    # Work-precision mode: build RK4 and RKCK45 once at NT=32768 and run the
    # dt/tolerance sweeps ("Lorenz.exe 32768 wp") instead of the N sweep.
    [switch]$Wp
)

$ErrorActionPreference = 'Stop'

# Run from the repo root regardless of the caller's working directory
Push-Location (Join-Path $PSScriptRoot '..\..')

function Enter-VsEnvironment {
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        return  # already in a developer shell
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

function Build-Project {
    # Mirrors GPU_ODE_MPGOS/Makefile. --gpu-architecture=native targets the
    # local GPU (sm_70 was dropped in CUDA 13). MSVC does not support
    # -std=c++11, so c++17 is used on Windows.
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

Enter-VsEnvironment

if ($Wp) {
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"
    $content[16] = "const int NT = 32768;"

    # RK4 build -> fixed-dt sweep
    $content[14] = "#define SOLVER RK4"
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"
    Build-Project
    & "GPU_ODE_MPGOS\Lorenz.exe" 32768 wp
    if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RK4 wp) failed with exit code $LASTEXITCODE" }

    # RKCK45 build -> adaptive-tolerance sweep
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"
    $content[14] = "#define SOLVER RKCK45"
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"
    Build-Project
    & "GPU_ODE_MPGOS\Lorenz.exe" 32768 wp
    if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RKCK45 wp) failed with exit code $LASTEXITCODE" }

    Pop-Location
    return
}

$a = 8

while ($a -le $MaxA) {
    Write-Host $a

    # Read the file content
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"

    # Replace line 15 with RK4 solver definition
    $content[14] = "#define SOLVER RK4"

    # Replace line 17 with NT value
    $content[16] = "const int NT = $a;"

    # Write back to file
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"

    # Build and run with RK4
    Build-Project
    & "GPU_ODE_MPGOS\Lorenz.exe" $a
    if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RK4) failed with exit code $LASTEXITCODE" }

    # Read the file content again
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"

    # Replace line 15 with RKCK45 solver definition
    $content[14] = "#define SOLVER RKCK45"

    # Write back to file
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"

    # Build and run with RKCK45
    Build-Project
    & "GPU_ODE_MPGOS\Lorenz.exe" $a
    if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RKCK45) failed with exit code $LASTEXITCODE" }

    # Increment the value
    $a = $a * 4
}

Pop-Location
