# ============================================================
# Face Emotion Recognizer — Flask Backend
# ============================================================
# Endpoint : POST /predict
# Input    : multipart/form-data  →  file=<image>
# Output   : JSON { "emotion": str, "confidence": float,
#                   "all_emotions": {label: prob, ...} }
# ============================================================

import os
import io
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model   # type: ignore

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
# Labels must match the order used when the model was trained.
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# Expected input size for the CNN (height, width)
IMG_SIZE = (48, 48)

# Path to the pre-trained Keras model file
MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")

# OpenCV ships the Haar Cascade XML inside the cv2 package data folder
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Load model & cascade once at startup ───────────────────
logger.info("Loading emotion model from '%s' …", MODEL_PATH)
model = load_model(MODEL_PATH)
logger.info("Model loaded.  Input shape: %s", model.input_shape)

logger.info("Loading Haar Cascade from '%s' …", CASCADE_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar Cascade from: {CASCADE_PATH}")
logger.info("Haar Cascade loaded.")

# ── Flask app ──────────────────────────────────────────────
app = Flask(__name__)


# ── Helper: decode uploaded bytes → BGR numpy array ───────
def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Convert raw uploaded bytes into an OpenCV BGR image array."""
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file.")
    return img


# ── Helper: detect the largest face ───────────────────────
def _detect_largest_face(bgr_img: np.ndarray):
    """
    Run Haar Cascade face detection and return the ROI (x, y, w, h)
    of the largest detected face, or None if no face is found.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # detectMultiScale parameters:
    #   scaleFactor  – how much the image is shrunk at each scale (1.1 = 10 %)
    #   minNeighbors – higher → fewer false positives
    #   minSize      – ignore tiny regions
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None, gray

    # Pick the bounding box with the largest area (w * h)
    largest = max(faces, key=lambda rect: rect[2] * rect[3])
    return largest, gray


# ── Helper: preprocess face ROI for model input ────────────
def _preprocess_face(gray_img: np.ndarray, roi: tuple) -> np.ndarray:
    """
    Crop the face region from the grayscale image, resize to IMG_SIZE,
    normalise pixel values to [0, 1], and reshape into the batch tensor
    expected by the model: (1, H, W, 1).
    """
    x, y, w, h = roi

    # Crop face region from grayscale image
    face_roi = gray_img[y : y + h, x : x + w]

    # Resize to the model's expected input dimensions (48 × 48)
    face_resized = cv2.resize(face_roi, IMG_SIZE)

    # Normalise pixel values from [0, 255] → [0.0, 1.0]
    face_normalised = face_resized.astype("float32") / 255.0

    # Add channel dimension  →  (48, 48, 1)
    face_with_channel = np.expand_dims(face_normalised, axis=-1)

    # Add batch dimension  →  (1, 48, 48, 1)
    face_batch = np.expand_dims(face_with_channel, axis=0)

    return face_batch


# ── /predict endpoint ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    -------------
    Form field : file  (image/jpeg, image/png, …)

    Success response (HTTP 200):
    {
        "emotion":      "Happy",
        "confidence":   0.9421,
        "all_emotions": {
            "Angry": 0.0031, "Disgust": 0.0008, "Fear": 0.0047,
            "Happy": 0.9421, "Sad": 0.0211, "Surprise": 0.0263,
            "Neutral": 0.0019
        }
    }

    Error responses (HTTP 4xx / 5xx):
    { "error": "<description>" }
    """

    # ── 1. Validate request ────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Send an image under the key 'file'."}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename — please attach an actual image file."}), 400

    # ── 2. Read uploaded bytes ─────────────────────────────
    file_bytes = uploaded_file.read()
    if len(file_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    # ── 3. Decode image ────────────────────────────────────
    try:
        bgr_img = _decode_image(file_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    # ── 4. Detect face with Haar Cascade ──────────────────
    roi, gray_img = _detect_largest_face(bgr_img)
    if roi is None:
        return jsonify({"error": "No face detected in the image. Please use a clear frontal-face photo."}), 422

    # ── 5. Preprocess face ROI ─────────────────────────────
    face_batch = _preprocess_face(gray_img, roi)

    # ── 6. Model inference ─────────────────────────────────
    # model.predict() returns a 2-D array: [[p0, p1, …, p6]]
    predictions = model.predict(face_batch, verbose=0)[0]  # shape: (7,)

    # ── 7. Extract top emotion & confidence ───────────────
    top_idx = int(np.argmax(predictions))
    top_emotion = EMOTION_LABELS[top_idx]
    top_confidence = float(predictions[top_idx])

    # Build a full probability mapping (rounded to 4 decimal places)
    all_emotions = {
        label: round(float(prob), 4)
        for label, prob in zip(EMOTION_LABELS, predictions)
    }

    logger.info(
        "Prediction → %s (%.1f%%) | file='%s'",
        top_emotion,
        top_confidence * 100,
        uploaded_file.filename,
    )

    # ── 8. Return JSON response ────────────────────────────
    return jsonify(
        {
            "emotion": top_emotion,
            "confidence": round(top_confidence, 4),
            "all_emotions": all_emotions,
        }
    ), 200


# ── Health check ───────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe — returns HTTP 200 when the server is up."""
    return jsonify({"status": "ok", "model": MODEL_PATH}), 200


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    # debug=False is intentional — never run debug=True in production
    app.run(host="0.0.0.0", port=5000, debug=False)
