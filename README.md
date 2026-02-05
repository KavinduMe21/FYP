# Rugby Knock-On Detection System

Final Year Project - Ball detection and tracking in rugby footage using Computer Vision and Machine Learning.

## Technologies Used

- **YOLOv8 (Ultralytics):** Deep learning model for object detection
- **FastAPI:** Modern web framework for building APIs
- **OpenCV (opencv-python):** Video processing, motion detection, and image filtering
- **Scikit-learn:** Support Vector Machine (SVM) classifier
- **NumPy:** Numerical array operations for image data
- **Joblib:** Model serialization and loading

## How to Run

### 1. Collect Training Data
Process videos in the `Dataset` folder to create positive (ball) and negative (background/player) samples:
```bash
python run.py --mode collect
```

### 2. Train Model
```bash
python run.py --mode train
```

### 3. Run Detection
```bash
python run.py --mode detect
```

### 4. Start Web API
```bash
uvicorn api.ballDetect:app --reload
```

Then open `frontend/index.html` in your browser to use the web interface.

## Installation

```bash
# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows

# Install dependencies
pip install ultralytics fastapi uvicorn opencv-python numpy scikit-learn joblib python-multipart
```

## Repository

https://github.com/KavinduMe21/FYP
