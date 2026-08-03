@echo off
setlocal enabledelayedexpansion

REM Fixed 2..4 sweep; no arguments.
set datadir=SDE

if exist "data\%datadir%\CRN\" rmdir /s /q "data\%datadir%\CRN"
mkdir "data\%datadir%\CRN"
if exist "data\CPU\%datadir%\CRN\" rmdir /s /q "data\CPU\%datadir%\CRN"
mkdir "data\CPU\%datadir%\CRN"

set a=2
:loop
if %a% gtr 4 goto end

echo No. of trajectories = %a%
julia --threads=16 --project=GPU_ODE_Julia GPU_ODE_Julia\sde_examples\bench_crn_model.jl %a%
if errorlevel 1 exit /b 1

set /a a=%a%*2
goto loop

:end
endlocal
