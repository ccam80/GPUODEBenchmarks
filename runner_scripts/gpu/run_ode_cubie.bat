@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_CUBIE\venv\Scripts\activate.bat

REM The venv is shared with the MLIR suite; cubie picks its backend from this at import time.
set CUBIE_CUDA_BACKEND=numba-cuda

if /i "%ANALYSIS%"=="states" (
    REM -n ^(when set^) overrides the states-sweep ensemble size.
    set "STATES_ARG=states"
    if not "%NMAX%"=="16777216" set "STATES_ARG=states:%NMAX%"
    python GPU_ODE_CUBIE\bench_cubie.py "!STATES_ARG!" "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_CUBIE\bench_cubie.py wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

REM The whole ascending N sweep runs in one process on kernels compiled once.
set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
echo N sweep = !NLIST_CSV!
python GPU_ODE_CUBIE\bench_cubie.py "!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1

call deactivate
endlocal
