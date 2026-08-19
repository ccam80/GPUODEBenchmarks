@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

REM The julia driver runs one process per leg with parallel compiles.
set "ALGO_LIST="
for /f "usebackq delims=" %%g in (`python runner_scripts\algorithms.py julia "%ALGORITHM%"`) do (
    set "ALGO_LIST=!ALGO_LIST! %%g"
)
if "!ALGO_LIST!"=="" (
    echo Julia ^(DiffEqGPU kernel path^) runs none of the requested algorithms; skipping.
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="warm" (
    REM Package precompilation runs in parallel; DiffEqGPU kernels do not persist.
    julia --project=. -e "using Pkg; Pkg.precompile()"
    if errorlevel 1 exit /b 1
    echo DiffEqGPU kernels recompile per process; the julia driver overlaps those compiles.
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="states" (
    REM Parallel compiles, serialized GPU sections; -n overrides the ensemble size.
    if not "%NMAX%"=="16777216" (
        python runner_scripts\gpu\julia_driver.py states "%ALGORITHM%" "%NMAX%"
    ) else (
        python runner_scripts\gpu\julia_driver.py states "%ALGORITHM%"
    )
    if errorlevel 1 exit /b 1
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="work-precision" (
    python runner_scripts\gpu\julia_driver.py wp "%ALGORITHM%" "%PROBLEM%"
    if errorlevel 1 exit /b 1
    endlocal
    exit /b 0
)

set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
echo N sweep = !NLIST_CSV!
python runner_scripts\gpu\julia_driver.py performance "!NLIST_CSV!" "%ALGORITHM%" "%PROBLEM%"
if errorlevel 1 exit /b 1

endlocal
