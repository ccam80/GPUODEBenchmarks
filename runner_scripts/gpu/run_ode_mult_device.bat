@echo off
setlocal enabledelayedexpansion

REM Backend is an extra axis here, so this one does not use the shared parser.
set NMAX=16777216
set BACKEND=

:parse_loop
if "%~1"=="" goto parse_done
if /i "%~1"=="-n" (
    set NMAX=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--nmax" (
    set NMAX=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-b" (
    set BACKEND=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--backend" (
    set BACKEND=%~2
    shift
    shift
    goto parse_loop
)
echo Unknown option %~1
exit /b 1
:parse_done
if "%BACKEND%"=="" (
    echo Usage: %~nx0 --backend ^<backend^> [-n nmax]
    exit /b 1
)

REM Clear this run's data directory.
if exist "data\devices\%BACKEND%\" rmdir /s /q "data\devices\%BACKEND%"
mkdir "data\devices\%BACKEND%" 2>nul

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
julia --project=GPU_ODE_Julia GPU_ODE_Julia\bench_multi_device.jl %a% %BACKEND%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
endlocal
