@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1
set path=CPU

REM Create or clear data directory
if not exist "data\%path%\" mkdir "data\%path%"
del /q "data\%path%\*" 2>nul

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo %a%
julia --threads=16 --project=GPU_ODE_Julia GPU_ODE_Julia\bench_cpu.jl %a%

REM Increment the value
set /a a=%a%*4
goto loop

:end
endlocal
