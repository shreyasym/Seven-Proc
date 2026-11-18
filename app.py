"""Local web interface for seven-segment meter OCR."""
from __future__ import annotations

import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from sevenseg.pipeline import read_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 10


@app.get("/")
def index():
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify(error="The upload is too large. Keep the total upload under 32 MB."), 413


@app.post("/api/detect")
def detect():
    uploads = request.files.getlist("images")
    if not uploads or all(not upload.filename for upload in uploads):
        return jsonify(error="Choose at least one meter image to process."), 400
    uploads = [upload for upload in uploads if upload.filename]
    if len(uploads) > MAX_IMAGES:
        return jsonify(error=f"Upload no more than {MAX_IMAGES} images at a time."), 400

    results = []
    for upload in uploads:
        filename = secure_filename(upload.filename) or "meter-image"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify(error=f"{filename}: use a JPG, PNG, WEBP, or BMP image."), 400

        contents = upload.stream.read(MAX_IMAGE_BYTES + 1)
        if len(contents) > MAX_IMAGE_BYTES:
            return jsonify(error=f"{filename}: the image must be smaller than 10 MB."), 400

        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return jsonify(error=f"{filename}: this file could not be decoded as an image."), 400

        started = time.perf_counter()
        reading = read_image(image)
        results.append({
            "filename": filename,
            "detected_reading": reading["detected_reading"],
            "raw_text": reading["raw_text"],
            "confidence_score": reading["confidence_score"],
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 1),
        })

    return jsonify(results=results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
