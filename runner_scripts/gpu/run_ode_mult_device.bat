@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1
set backend=%2

REM Create or clear data directory
if exist "data\devices\%backend%\" (
    rmdir /s /q "data\devices\%backend%" 2>nul
    mkdir "data\devices\%backend%"
) else (
    if not exist "data\devices\" mkdir "data\devices"
    mkdir "data\devices\%backend%"
)

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo %a%
julia --project=GPU_ODE_Julia GPU_ODE_Julia\bench_multi_device.jl %a% %backend%

REM Increment the value
set /a a=%a%*4
goto loop

:end
endlocal
