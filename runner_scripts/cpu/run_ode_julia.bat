@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1
if /i not "%ANALYSIS%"=="performance" (
    echo CPU Julia supports -a performance only
    exit /b 1
)

REM Clear this run's data directory.
if exist "data\CPU\" rmdir /s /q "data\CPU"
mkdir "data\CPU" 2>nul

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
julia --threads=16 --project=GPU_ODE_Julia GPU_ODE_Julia\bench_cpu.jl %a%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
endlocal
