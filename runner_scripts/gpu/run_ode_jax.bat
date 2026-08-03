@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1

call GPU_ODE_JAX\venv\Scripts\activate.bat

set XLA_PYTHON_CLIENT_PREALLOCATE=false

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_JAX\bench_diffrax.py 32768 wp "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
python GPU_ODE_JAX\bench_diffrax.py %a% "%ALGORITHM%"
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
call deactivate
endlocal
