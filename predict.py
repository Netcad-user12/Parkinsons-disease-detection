import torch
import cv2
import numpy as np
from model import CNN_LSTM_Model

# -------------------------------
# Load Model
# -------------------------------
def load_model(path):
    model = CNN_LSTM_Model()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# -------------------------------
# Preprocess (for hybrid model)
# -------------------------------
def preprocess_image(img_path, seq_len=4):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    step = w // seq_len

    patches = []
    for i in range(seq_len):
        patch = img[:, i*step:(i+1)*step]
        patch = cv2.resize(patch, (28, 28))
        patch = patch / 255.0
        patch = np.expand_dims(patch, axis=0)
        patches.append(patch)

    sequence = np.array(patches)
    sequence = np.expand_dims(sequence, axis=0)

    return torch.tensor(sequence, dtype=torch.float32)

# -------------------------------
# Predict
# -------------------------------
def predict(img_path, model):
    data = preprocess_image(img_path)

    with torch.no_grad():
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()