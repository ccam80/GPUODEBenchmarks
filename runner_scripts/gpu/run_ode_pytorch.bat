@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_PyTorch\venv\Scripts\activate.bat

REM Fixed-step only: torchdiffeq adaptive solvers are incompatible with torch.vmap.
if /i "%ANALYSIS%"=="warm" (
    set "NLIST_CSV=!NLIST: =,!"
    if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
    python GPU_ODE_PyTorch\bench_torchdiffeq.py "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="states" (
    REM -n ^(when set^) is the state-count list or ceiling.
    set "STATES_ARG=states"
    if not "!NMAX_RAW!"=="16777216" set "STATES_ARG=states:!NMAX_RAW!"
    python GPU_ODE_PyTorch\bench_torchdiffeq.py "!STATES_ARG!" "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_PyTorch\bench_torchdiffeq.py wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

REM The whole ascending N sweep runs in one process on kernels compiled once.
set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
echo N sweep = !NLIST_CSV!
python GPU_ODE_PyTorch\bench_torchdiffeq.py "!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1

call deactivate
endlocal
