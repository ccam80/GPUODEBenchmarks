@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_MYOKIT_CUDA\venv\Scripts\activate.bat

REM Myokit CUDA exposes float32 forward Euler only, so work-precision is fixed-step.
if /i "%ANALYSIS%"=="warm" (
    set "NLIST_CSV=!NLIST: =,!"
    if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="states" (
    REM -n ^(when set^) overrides the states-sweep ensemble size.
    set "STATES_ARG=states"
    if not "%NMAX%"=="16777216" set "STATES_ARG=states:%NMAX%"
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py "!STATES_ARG!" "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

REM The whole ascending N sweep runs in one process on kernels compiled once.
set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
echo N sweep = !NLIST_CSV!
python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1
python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py "!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1

call deactivate
endlocal
