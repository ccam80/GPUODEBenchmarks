@echo off
setlocal enabledelayedexpansion

REM Algorithm request (issue #29): forwarded to the bench script, which runs
REM every supported algorithm for "all" and skips cleanly when unsupported.
set "alg=%~2"
if "%alg%"=="" set "alg=all"

REM Work-precision mode: `run_ode_julia.bat wp` sweeps dt/tolerance at N=32768.
if /i "%~1"=="wp" (
    julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl 32768 wp %alg%
    if errorlevel 1 exit /b 1
    endlocal
    exit /b 0
)

set a=8
set max_a=%1

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo %a%
julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl %a% %alg%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
endlocal
