#!/bin/bash
# ========================================================
#   DeepPCB Inspect AI Dashboard - Unix Launcher
# ========================================================

echo "========================================================"
echo "  Starting DeepPCB Inspect AI Dashboard"
echo "  Machine Learning Minor Degree Final Project"
echo "========================================================"
echo ""

if [ -f "weights/best.pt" ]; then
    echo "[OK] Detected weights/best.pt! Running with live PyTorch YOLOv8."
else
    echo "[INFO] weights/best.pt not found yet. Running in Interactive Simulation Mode."
fi

echo ""
echo "Starting Web Dashboard on http://127.0.0.1:8000 ..."

# Attempt to open default browser
if which xdg-open > /dev/null; then
    xdg-open "http://127.0.0.1:8000" &
elif which open > /dev/null; then
    open "http://127.0.0.1:8000" &
fi

python3 app.py

