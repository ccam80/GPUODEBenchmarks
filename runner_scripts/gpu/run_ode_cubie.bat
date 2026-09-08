@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_CUBIE\venv\Scripts\activate.bat

REM The suite holds ~100 kernels per system; the default LRU cap of 10 evicts them.
set CUBIE_MAX_CACHE_ENTRIES=0
set "BENCH=python GPU_ODE_CUBIE\bench_cubie.py"

if /i "%ANALYSIS%"=="optimize" (
    %BENCH% optimize "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="warm" (
    set "NLIST_CSV=!NLIST: =,!"
    if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
    %BENCH% "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="states" (
    %BENCH% states "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="work-precision" (
    %BENCH% optimize "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    %BENCH% wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

REM Optimize, warm the tuned kernels, then walk the ascending N sweep in one process.
set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
echo N sweep = !NLIST_CSV!
%BENCH% optimize "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1
%BENCH% "warm:!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1
%BENCH% "!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1

call deactivate
endlocal
