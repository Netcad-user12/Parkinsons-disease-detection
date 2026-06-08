import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=4):
        self.root_dir = root_dir
        self.seq_len = seq_len

        self.data = []
        self.labels = []

        # ✅ Correct way to get class folders
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # ✅ Load image paths + labels
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)

            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)

                # Ignore non-image files
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(file_path)
                    self.labels.append(label)

        # ❗ Safety check
        if len(self.data) == 0:
            raise ValueError("Dataset is empty! Check your data folder.")

    def __len__(self):
        return len(self.data)

    # ✅ Split image into sequence patches
    def split_image(self, img):
        h, w = img.shape
        patches = []

        step = w // self.seq_len

        for i in range(self.seq_len):
            patch = img[:, i*step:(i+1)*step]

            # Handle edge case if width not divisible
            if patch.shape[1] == 0:
                continue

            patch = cv2.resize(patch, (28, 28))
            patches.append(patch)

        return patches

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # ❗ Handle broken images
        if img is None:
            raise ValueError(f"Error reading image: {img_path}")

        patches = self.split_image(img)

        sequence = []
        for p in patches:
            p = p / 255.0
            p = np.expand_dims(p, axis=0)  # (1, 28, 28)
            sequence.append(p)

        # Ensure fixed sequence length
        while len(sequence) < self.seq_len:
            sequence.append(sequence[-1])

        sequence = np.array(sequence)  # (seq_len, 1, 28, 28)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)