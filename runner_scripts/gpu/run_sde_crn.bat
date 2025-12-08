@echo off
setlocal enabledelayedexpansion

set a=2
set max_a=4
set path=SDE

REM Create or clear data directories
if exist "data\%path%\CRN\" (
    rmdir /s /q "data\%path%\CRN" 2>nul
)
mkdir "data\%path%\CRN"

if exist "data\CPU\%path%\CRN\" (
    rmdir /s /q "data\CPU\%path%\CRN" 2>nul
)
mkdir "data\CPU\%path%\CRN"

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo %a%
julia --threads=16 --project=GPU_ODE_Julia GPU_ODE_Julia\sde_examples\bench_crn_model.jl %a%

REM Increment the value
set /a a=%a%*2
goto loop

:end
endlocal
