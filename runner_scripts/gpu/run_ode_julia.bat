@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo %a%
julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl %a%

REM Increment the value
set /a a=%a%*4
goto loop

:end
endlocal
