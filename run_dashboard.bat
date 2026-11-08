@echo off
echo ========================================================
echo   Starting DeepPCB Inspect AI Dashboard
echo   Machine Learning Minor Degree Final Project
echo ========================================================
echo.
echo Checking weights...
if exist "weights\best.pt" (
    echo [OK] Detected weights\best.pt! Running with live PyTorch YOLOv8.
) else (
    echo [INFO] weights\best.pt not found yet. Running in Interactive Simulation Mode.
    echo        Place your trained best.pt into the "weights\" folder anytime.
)
echo.
echo Starting Web Dashboard on http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
python app.py
pause
