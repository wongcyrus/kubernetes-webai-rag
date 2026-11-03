@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Missing portable Python. Extract the application archive first.
  pause
  exit /b 1
)
"runtime\python.exe" "serve.py"
if errorlevel 1 (
  pause
  exit /b 1
)
exit /b 0
