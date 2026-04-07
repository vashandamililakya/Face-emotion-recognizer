# ============================================================
# Face Emotion Recognizer — Flask Backend (DeepFace version)
# ============================================================
# Endpoint : POST /predict
# Input    : multipart/form-data  →  file=<image>
# Output   : JSON { "emotion": str, "confidence": float,
#                   "all_emotions": {label: prob, ...} }
# ============================================================

import os
import logging
import tempfile

import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────
app = Flask(__name__)


# ── /predict endpoint ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    -------------
    Form field : file  (image/jpeg, image/png, …)

    Success response (HTTP 200):
    {
        "emotion":      "happy",
        "confidence":   0.9421,
        "all_emotions": {
            "angry": 0.0031, "disgust": 0.0008, "fear": 0.0047,
            "happy": 0.9421, "sad": 0.0211, "surprise": 0.0263,
            "neutral": 0.0019
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

    # ── 3. Save to a temp file (DeepFace needs a file path) ─
    suffix = os.path.splitext(uploaded_file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # ── 4. Run DeepFace emotion analysis ──────────────────
    try:
        results = DeepFace.analyze(
            img_path=tmp_path,
            actions=["emotion"],
            enforce_detection=True,   # raises if no face found
            silent=True,
        )
    except ValueError as exc:
        # DeepFace raises ValueError when no face is detected
        os.unlink(tmp_path)
        logger.warning("No face detected: %s", exc)
        return jsonify({"error": "No face detected in the image. Please use a clear frontal-face photo."}), 422
    except Exception as exc:
        os.unlink(tmp_path)
        logger.error("DeepFace error: %s", exc)
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 500
    finally:
        # Clean up temp file if it still exists
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── 5. Extract results ─────────────────────────────────
    # DeepFace returns a list when analyze() processes one image
    face_data = results[0] if isinstance(results, list) else results

    top_emotion   = face_data["dominant_emotion"]          # e.g. "happy"
    emotion_probs = face_data["emotion"]                   # dict {label: score 0-100}

    # Convert scores from 0-100 range → 0.0-1.0 and round
    all_emotions = {
        label: round(float(score) / 100.0, 4)
        for label, score in emotion_probs.items()
    }

    top_confidence = all_emotions.get(top_emotion, 0.0)

    logger.info(
        "Prediction → %s (%.1f%%) | file='%s'",
        top_emotion,
        top_confidence * 100,
        uploaded_file.filename,
    )

    # ── 6. Return JSON response ────────────────────────────
    return jsonify(
        {
            "emotion":      top_emotion,
            "confidence":   top_confidence,
            "all_emotions": all_emotions,
        }
    ), 200


# ── Health check ───────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe — returns HTTP 200 when the server is up."""
    return jsonify({"status": "ok", "backend": "DeepFace"}), 200


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
