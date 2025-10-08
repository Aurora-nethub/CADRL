@echo off
REM Minimal runner for CADRL on Windows (PowerShell or CMD)
REM Usage:
REM    run.bat --run-name myrun --lr 0.001 --epochs 10

set PYTHON=%PYTHON%
if "%PYTHON%"=="" set PYTHON=python

%PYTHON% "%~dp0\main.py" %*
