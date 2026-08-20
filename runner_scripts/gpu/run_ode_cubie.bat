@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_CUBIE\venv\Scripts\activate.bat

REM The venv is shared with the MLIR suite; cubie picks its backend from this at import time.
set CUBIE_CUDA_BACKEND=numba-cuda
REM The suite holds ~100 kernels per system; the default LRU cap of 10 evicts them.
set CUBIE_MAX_CACHE_ENTRIES=0

if /i "%ANALYSIS%"=="warm" (
    set "NLIST_CSV=!NLIST: =,!"
    if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
    python GPU_ODE_CUBIE\bench_cubie.py "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="states" (
    python GPU_ODE_CUBIE\bench_cubie.py states "%ALGORITHM%"
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
python GPU_ODE_CUBIE\bench_cubie.py "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1
python GPU_ODE_CUBIE\bench_cubie.py "!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1

call deactivate
endlocal
