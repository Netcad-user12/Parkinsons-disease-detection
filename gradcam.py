# =========================
# IMPORTS
# =========================
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

IMG_SIZE = 128

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model.h5")

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, np.expand_dims(img, axis=0)

# =========================
# GRAD-CAM FUNCTION
# =========================
def get_gradcam(model, img_array, last_conv_layer_name="last_conv"):

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy(), predictions.numpy()[0][0]

# =========================
# DISPLAY RESULT
# =========================
def show_result(img_path):

    original_img, img_array = preprocess(img_path)

    heatmap, prediction = get_gradcam(model, img_array)

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img

    # =========================
    # EXPLANATION LOGIC
    # =========================
    print("\n========================")

    if prediction > 0.5:
        print("⚠️ Parkinson's Likely Detected")

        print("""
Explainable AI Insight:
- Red/Yellow regions = model focused here
- These areas may contain shaky or irregular strokes

Interpretation:
- Possible tremor patterns detected
        """)

    else:
        print("✅ Healthy Pattern Detected")

        print("""
Explainable AI Insight:
- Focus is spread across smooth strokes
- No strong irregular regions

Interpretation:
- Writing appears stable
        """)

    print("⚠️ This is AI-based explanation, not a medical diagnosis.")

    # =========================
    # SHOW IMAGES
    # =========================
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(superimposed.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.show()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    path = input("Enter image path: ")
    show_result(path)