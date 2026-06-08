import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# -------------------------------
# Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Dataset
# -------------------------------
from dataset import ImageSequenceDataset

dataset = ImageSequenceDataset("data", seq_len=4)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)

# -------------------------------
# Load Model
# -------------------------------
from model import CNN_LSTM_Model

model = CNN_LSTM_Model().to(device)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()

# -------------------------------
# Evaluation
# -------------------------------
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for sequences, labels in loader:

        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)

        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(probs, 1)

        # Parkinson probability
        all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------------
# Metrics
# -------------------------------
accuracy = accuracy_score(all_labels, all_preds)

precision = precision_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

auc_score = roc_auc_score(
    all_labels,
    all_probs
)

# -------------------------------
# Model Accuracy Table
# -------------------------------
print("\n==============================================")
print("            MODEL ACCURACY TABLE")
print("==============================================")

print("{:<25} {:<15}".format("Metric", "Value"))
print("-" * 40)

print("{:<25} {:<15}".format(
    "Accuracy",
    f"{accuracy*100:.2f}%"
))

print("{:<25} {:<15}".format(
    "Precision (Macro Avg)",
    f"{precision:.4f}"
))

print("{:<25} {:<15}".format(
    "Recall (Macro Avg)",
    f"{recall:.4f}"
))

print("{:<25} {:<15}".format(
    "F1-Score (Macro Avg)",
    f"{f1:.4f}"
))

print("{:<25} {:<15}".format(
    "AUC (ROC)",
    f"{auc_score:.4f}"
))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Healthy", "Parkinson"],
    yticklabels=["Healthy", "Parkinson"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -------------------------------
# ROC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probs
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

plt.show()