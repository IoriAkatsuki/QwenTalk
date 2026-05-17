@echo off
setlocal

:: QwenTalk GUI Launcher
:: This script starts the main application.

:: --- Configuration ---
:: Name of your Conda environment where the dependencies are installed.
set CONDA_ENV_NAME=openvino


:: --- Pre-run Checks ---
echo Checking for required Conda environment and dependencies...

:: 1. Check for Conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found. Please install Miniconda or Anaconda.
    pause
    exit /b 1
)

:: 2. Check for the environment
conda env list | findstr /B "%CONDA_ENV_NAME% " >nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda environment '%CONDA_ENV_NAME%' not found.
    echo Please create it and install dependencies by running:
    echo conda create -n %CONDA_ENV_NAME% python=3.10 -y
    echo conda run -n %CONDA_ENV_NAME% pip install -r requirements.txt
    pause
    exit /b 1
)

:: 3. Check for the model directory
if not exist "Qwen3-8B-nf4-ov\openvino_model.xml" (
    echo [ERROR] Model file 'Qwen3-8B-nf4-ov\openvino_model.xml' not found.
    echo Please ensure the OpenVINO model is correctly placed in the 'Qwen3-8B-nf4-ov' directory.
    pause
    exit /b 1
)


:: --- Launch the Application ---
echo All checks passed. Starting the QwenTalk GUI...

call conda run -n %CONDA_ENV_NAME% python QwenTalkGUI.py

if %errorlevel% neq 0 (
    echo [ERROR] The application exited with an error.
    pause
)

echo.
echo Application closed.
endlocal