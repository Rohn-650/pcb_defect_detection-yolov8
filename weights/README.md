# Model Weights (`weights/`)

This directory contains the fine-tuned YOLOv8s checkpoint trained on the industrial **DeepPCB dataset**.

## 📦 Checkpoint Details

| Attribute | Specification |
| :--- | :--- |
| **Filename** | `best.pt` |
| **Base Architecture** | YOLOv8s (`yolov8s.pt`) |
| **Model Size** | 22.5 MB (22,530,154 bytes) |
| **Parameters** | 11,127,906 (11.13M) |
| **Computational Complexity** | 28.4 GFLOPs (72 fused layers) |
| **Trained Epochs** | 100 epochs |
| **Input Resolution** | 640 × 640 px |
| **Test Performance** | **98.40% mAP@50** \| **96.82% Precision** \| **96.47% Recall** |
| **Inference Latency** | ~12.4 ms (Tesla T4 GPU) / ~45 ms (Intel Core i7 CPU) |

## 🏷️ Defect Classes (6 Categories)
1. `missing_hole` (Missing Hole / Un-drilled via) - Class ID: 0
2. `mouse_bite` (Mouse Bite / Edge Conductor Notch) - Class ID: 1
3. `open_circuit` (Open Circuit / Trace Discontinuity) - Class ID: 2
4. `short` (Short Circuit / Accidental Metallic Bridge) - Class ID: 3
5. `spur` (Spur / Metallic Protrusion) - Class ID: 4
6. `spurious_copper` (Spurious Copper / Isolated Residue) - Class ID: 5

## 🔄 Using Custom / Retrained Weights
To swap or update the model weights:
1. Export your trained PyTorch checkpoint as `best.pt`.
2. Replace `weights/best.pt`.
3. Restart or reload the dashboard at `http://127.0.0.1:8000`. The application will automatically detect the updated model.

