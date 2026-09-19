@echo off
setlocal
cd /d "%~dp0"
title PanelForge Geometry - Repair

echo This will rebuild PanelForge's LOCAL Python environment only.
echo Your CBZ files, source images and output folders are not touched.
echo.

if exist ".venv" (
    echo Removing incomplete local environment...
    rmdir /s /q ".venv"
)

echo.
echo Rebuilding...
call run.bat
