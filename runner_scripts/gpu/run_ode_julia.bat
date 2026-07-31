@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1

if /i "%ANALYSIS%"=="work-precision" (
    julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl 32768 wp
    if errorlevel 1 exit /b 1
    endlocal
    exit /b 0
)

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl %a%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
endlocal
