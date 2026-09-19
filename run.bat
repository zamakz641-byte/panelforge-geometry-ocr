@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title PanelForge Geometry + OCR v1.6.2

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

rem ------------------------------------------------------------
rem 1) Create the local environment only if it does not exist.
rem ------------------------------------------------------------
if not exist "%PY%" (
    echo.
    echo [PanelForge] First launch - creating local Python environment...
    echo [PanelForge] Do not close this window during setup.
    echo.

    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 goto :no_python
        python -m venv "%VENV%"
    )

    if errorlevel 1 goto :venv_fail
)

rem ------------------------------------------------------------
rem 2) Make sure pip itself exists, even after an interrupted setup.
rem ------------------------------------------------------------
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [PanelForge] Repairing pip...
    "%PY%" -m ensurepip --upgrade
    if errorlevel 1 goto :deps_fail
)

rem ------------------------------------------------------------
rem 3) REAL dependency check. A .venv folder alone is NOT enough.
rem    If setup was interrupted, the next launch resumes installation.
rem ------------------------------------------------------------
call :check_dependencies
if errorlevel 1 (
    echo.
    echo [PanelForge] Missing or incomplete dependencies detected.
    echo [PanelForge] Installing / repairing them now...
    echo [PanelForge] Keep this window open until installation finishes.
    echo.

    "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :deps_fail

    call :check_dependencies
    if errorlevel 1 goto :deps_fail
)

rem ------------------------------------------------------------
rem 4) Start the app only after dependencies are confirmed.
rem ------------------------------------------------------------
echo [PanelForge] Dependencies OK. Starting application...
"%PY%" app.py
if errorlevel 1 goto :app_fail
exit /b 0

:check_dependencies
"%PY%" -c "import webview, cv2, numpy; from PIL import Image" >nul 2>nul
exit /b %errorlevel%

:no_python
echo.
echo [ERROR] Python 3 was not found.
echo Install Python 3, then launch run.bat again.
pause
exit /b 1

:venv_fail
echo.
echo [ERROR] Could not create the local Python environment.
echo You can delete the .venv folder and launch run.bat again.
pause
exit /b 1

:deps_fail
echo.
echo [ERROR] PanelForge dependencies could not be installed completely.
echo Check your Internet connection, then simply launch run.bat again.
echo The launcher will retry and repair the existing .venv automatically.
pause
exit /b 1

:app_fail
echo.
echo [ERROR] Dependencies are installed, but PanelForge stopped with an error.
echo Copy the traceback shown above if you need to report the problem.
pause
exit /b 1
