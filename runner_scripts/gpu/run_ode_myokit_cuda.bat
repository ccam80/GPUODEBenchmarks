@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1

call GPU_ODE_MYOKIT_CUDA\venv\Scripts\activate.bat

REM Myokit CUDA exposes float32 forward Euler only, so work-precision is fixed-step.
if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py 32768 wp "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

for %%a in (!NLIST!) do (
    echo No. of trajectories = %%a
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py %%a "%ALGORITHM%"
    if !errorlevel! neq 0 exit /b 1
)

call deactivate
endlocal
