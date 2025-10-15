@echo off
REM Minimal runner for CADRL on Windows (PowerShell or CMD)

REM Usage:
REM    run.bat --mode train --run-name myrun
REM    run.bat --mode test  --run-name myrun

set PYTHON=%PYTHON%
if "%PYTHON%"=="" set PYTHON=python

%PYTHON% "%~dp0\main.py" %*
