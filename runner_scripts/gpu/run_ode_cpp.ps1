# PowerShell script for running C++ ODE benchmarks
param(
    [Parameter(Mandatory=$true)]
    [int]$MaxA
)

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
    Push-Location GPU_ODE_MPGOS
    nmake /f Makefile clean 2>$null
    if ($LASTEXITCODE -ne 0) {
        # If nmake fails, try make (for MinGW or similar)
        make clean
        make
    } else {
        nmake /f Makefile
    }
    Pop-Location
    
    & "GPU_ODE_MPGOS\Lorenz.exe" $a
    
    # Read the file content again
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"
    
    # Replace line 15 with RKCK45 solver definition
    $content[14] = "#define SOLVER RKCK45"
    
    # Write back to file
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"
    
    # Build and run with RKCK45
    Push-Location GPU_ODE_MPGOS
    nmake /f Makefile clean 2>$null
    if ($LASTEXITCODE -ne 0) {
        # If nmake fails, try make (for MinGW or similar)
        make clean
        make
    } else {
        nmake /f Makefile
    }
    Pop-Location
    
    & "GPU_ODE_MPGOS\Lorenz.exe" $a
    
    # Increment the value
    $a = $a * 4
}
