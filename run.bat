@echo off
REM Navigate to the directory where the batch file is located
cd /d %~dp0

REM Start the Python HTTP server on port 8000
start python -m http.server 8000

REM Wait a moment to ensure the server starts
timeout /t 2 >nul

REM Open the default web browser to the specified URL
start "" "http://localhost:8000/index.html"
