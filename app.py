import streamlit as st
import torch
import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# -------------------------------
# Import modules
# -------------------------------
from model import CNN_LSTM_Model
from predict import preprocess_image, predict
from gradcam_torch import GradCAM

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = CNN_LSTM_Model()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Grad-CAM Setup
# -------------------------------
target_layer = model.cnn[-1]
gradcam = GradCAM(model, target_layer)

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Parkinson's Detection (Hybrid CNN-LSTM)")
st.write("Upload a spiral or handwriting image")

uploaded_file = st.file_uploader(
    "Upload File",
    type=["jpg", "png", "jpeg", "zip"]
)

# -------------------------------
# Prediction + ZIP Extraction
# -------------------------------
if uploaded_file is not None:

    # -------------------------------
    # ZIP FILE
    # -------------------------------
    if uploaded_file.name.endswith(".zip"):

        # Save uploaded ZIP
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create extraction folder
        if not os.path.exists("extracted_files"):
            os.makedirs("extracted_files")

        # Extract ZIP
        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            zip_ref.extractall("extracted_files")

        st.success("✅ ZIP Extracted Successfully!")

        # -------------------------------
        # Find ALL files inside folders
        # -------------------------------
        files = []

        for root, dirs, filenames in os.walk("extracted_files"):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        st.write("📂 Extracted Files:")
        st.write(files)

        # -------------------------------
        # Process images
        # -------------------------------
        for file_path in files:

            if file_path.endswith((".jpg", ".png", ".jpeg")):

                # Show image
                st.image(
                    file_path,
                    caption=os.path.basename(file_path),
                    use_column_width=True
                )

                st.write("🔍 Analyzing...")

                # Prediction
                result, confidence = predict(file_path, model)

                if result == 0:
                    st.success(f"✅ Healthy ({confidence*100:.2f}%)")
                else:
                    st.error(f"⚠️ Parkinson's Detected ({confidence*100:.2f}%)")

                # -------------------------------
                # Grad-CAM
                # -------------------------------
                input_tensor = preprocess_image(file_path).to(device)

                heatmap = gradcam.generate(input_tensor)

                img = cv2.imread(file_path)

                heatmap = cv2.resize(
                    heatmap,
                    (img.shape[1], img.shape[0])
                )

                heatmap = np.uint8(255 * heatmap)

                heatmap = cv2.applyColorMap(
                    heatmap,
                    cv2.COLORMAP_JET
                )

                # Overlay
                overlay = heatmap * 0.4 + img
                overlay = np.clip(
                    overlay,
                    0,
                    255
                ).astype(np.uint8)

                st.image(
                    overlay,
                    caption=f"🔥 Grad-CAM Visualization - {os.path.basename(file_path)}",
                    use_column_width=True
                )

    # -------------------------------
    # SINGLE IMAGE FILE
    # -------------------------------
    else:

        temp_path = "temp.jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Show uploaded image
        st.image(
            temp_path,
            caption="Uploaded Image",
            use_column_width=True
        )

        st.write("🔍 Analyzing...")

        # Prediction
        result, confidence = predict(temp_path, model)

        if result == 0:
            st.success(f"✅ Healthy ({confidence*100:.2f}%)")
        else:
            st.error(f"⚠️ Parkinson's Detected ({confidence*100:.2f}%)")

        # -------------------------------
        # Grad-CAM
        # -------------------------------
        input_tensor = preprocess_image(temp_path).to(device)

        heatmap = gradcam.generate(input_tensor)

        img = cv2.imread(temp_path)

        heatmap = cv2.resize(
            heatmap,
            (img.shape[1], img.shape[0])
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # Overlay
        overlay = heatmap * 0.4 + img
        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(np.uint8)

        st.image(
            overlay,
            caption="🔥 Grad-CAM Visualization",
            use_column_width=True
        )

        # Cleanup
        os.remove(temp_path)

# -------------------------------
# Confusion Matrix + Metrics
# -------------------------------
st.write("---")
st.subheader("📊 Model Evaluation")

if st.button("Show Confusion Matrix"):

    try:
        from evaluate import true_labels, pred_labels

        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=["Healthy", "PD"],
            yticklabels=["Healthy", "PD"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        st.pyplot(fig)

        # F1 Score
        f1 = f1_score(true_labels, pred_labels)

        st.success(f"✅ F1 Score: {f1:.4f}")

        # Classification Report
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=["Healthy", "Parkinson"]
        )

        st.text("📄 Classification Report:")
        st.text(report)

    except:
        st.warning("⚠️ Run evaluate.py first to generate predictions.")