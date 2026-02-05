import cv2
import numpy as np
import os
import joblib
# cv2 has HOGDescriptor.

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, svm_model_path

def compute_hog_features(image):
    # Resize to fixed size
    image = cv2.resize(image, (64, 64))
    
    # WinSize, BlockSize, BlockStride, CellSize, NBin
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG descriptors
    h = hog.compute(image)
    return h.flatten()

def train():
    pos_dir = os.path.join(PROCESSED_DATA_DIR, 'positive')
    neg_dir = os.path.join(PROCESSED_DATA_DIR, 'negative')
    
    data = []
    labels = []
    
    print("Loading positive samples...")
    for filename in os.listdir(pos_dir):
        if filename.endswith(".jpg"):
            path = os.path.join(pos_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                features = compute_hog_features(img)
                data.append(features)
                labels.append(1) # 1 for Ball

    print(f"Loaded {len(labels)} positive samples.")

    print("Loading negative samples...")
    for filename in os.listdir(neg_dir):
        if filename.endswith(".jpg"):
            path = os.path.join(neg_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                features = compute_hog_features(img)
                data.append(features)
                labels.append(0) # 0 for Background

    print(f"Loaded {len(data) - labels.count(1)} negative samples.")
    
    if len(data) == 0:
        print("No data found! Please run collect_training_data.py first.")
        return

    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # Split
    print("Training SVM...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = LinearSVC(C=1.0, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save
    print(f"Saving model to {svm_model_path}")
    joblib.dump(model, svm_model_path)
    print("Done.")

if __name__ == "__main__":
    train()
