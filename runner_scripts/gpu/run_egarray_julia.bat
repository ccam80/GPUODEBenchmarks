@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1
set path=EnsembleGPUArray

REM Create or clear data directory
if exist "data\%path%\" (
    rmdir /s /q "data\%path%" 2>nul
    mkdir "data\%path%"
) else (
    mkdir "data\%path%"
)

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo %a%
julia --project=GPU_ODE_Julia GPU_ODE_Julia\bench_ensemblegpuarray.jl %a%

REM Increment the value
set /a a=%a%*4
goto loop

:end
endlocal
