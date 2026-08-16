@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

if /i "%ANALYSIS%"=="work-precision" (
    julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl 131072 wp "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    endlocal
    exit /b 0
)

for %%a in (!NLIST!) do (
    echo No. of trajectories = %%a
    julia --project=. GPU_ODE_Julia\bench_lorenz_gpu.jl %%a "%ALGORITHM%"
    if !errorlevel! neq 0 exit /b 1
)

endlocal
