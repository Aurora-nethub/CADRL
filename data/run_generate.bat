@echo off
set PYTHON=%PYTHON%
if "%PYTHON%"=="" set PYTHON=python

%PYTHON% "%~dp0generate_multi_sim_orca.py" %*
