import io

import cv2
import numpy as np

from app import app
from sevenseg.synth import render


def test_home_page_loads():
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Add your meter photos" in response.data
    assert b"Use camera" in response.data
    assert b"camera-preview" in response.data


def test_detect_reads_uploaded_meter_image():
    image = render("57901", np.random.default_rng(0))
    success, encoded = cv2.imencode(".png", image)
    assert success

    response = app.test_client().post(
        "/api/detect",
        data={"images": (io.BytesIO(encoded.tobytes()), "meter.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.get_json()["results"][0]
    assert result["filename"] == "meter.png"
    assert result["raw_text"] == "57901"
    assert result["detected_reading"] == 57901
    assert result["processing_time_ms"] >= 0


def test_detect_rejects_unsupported_and_invalid_images():
    client = app.test_client()
    unsupported = client.post(
        "/api/detect",
        data={"images": (io.BytesIO(b"not an image"), "meter.gif")},
        content_type="multipart/form-data",
    )
    invalid = client.post(
        "/api/detect",
        data={"images": (io.BytesIO(b"not an image"), "meter.png")},
        content_type="multipart/form-data",
    )

    assert unsupported.status_code == 400
    assert "JPG, PNG" in unsupported.get_json()["error"]
    assert invalid.status_code == 400
    assert "could not be decoded" in invalid.get_json()["error"]
