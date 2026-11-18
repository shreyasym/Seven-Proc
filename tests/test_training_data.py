import csv

import cv2
import numpy as np
import pytest

from annotate_meter_dataset import load_tasks
from sevenseg.crops import load_folder
from sevenseg.classifier import ConfidenceEnsemble, load_classifier
from sevenseg.pipeline import read_meter
from train_cnn import split_by_group


def test_annotation_tasks_require_consistent_labels_and_resume_completed_images(tmp_path):
    root = tmp_path / "archive"
    quality = "HQ_digital"
    image_dir = root / quality
    image_dir.mkdir(parents=True)
    (image_dir / "meter-a.jpg").write_bytes(b"image")
    (image_dir / "meter-b.jpg").write_bytes(b"image")
    rows = [
        {"image": "meter-a.jpg", "used_liter": "33", "cadran_1": "X", "cadran_2": "X", "cadran_3": "3", "cadran_4": "3"},
        {"image": "meter-b.jpg", "used_liter": "29", "cadran_1": "X", "cadran_2": "X", "cadran_3": "2", "cadran_4": "8"},
    ]
    with (root / f"{quality}.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "crops"
    tasks, skipped = load_tasks(root, output_dir, [quality])
    assert [task.filename for task in tasks] == ["meter-a.jpg"]
    assert tasks[0].labels == "33"
    assert skipped == 1

    for position in range(2):
        destination = output_dir / "3" / f"{quality}_meter-a_{position}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"crop")
    tasks, skipped = load_tasks(root, output_dir, [quality])
    assert tasks == []
    assert skipped == 1


def test_crop_loader_groups_digits_from_the_same_source_image(tmp_path):
    digit_dir = tmp_path / "7"
    digit_dir.mkdir()
    for filename in ("meter-a_0.png", "meter-a_1.png", "meter-b_0.png"):
        assert cv2.imwrite(str(digit_dir / filename), np.full((28, 28), 255, np.uint8))

    crops, labels, groups = load_folder(tmp_path)

    assert crops.shape == (3, 28, 28)
    assert labels.tolist() == [7, 7, 7]
    assert sorted(groups.tolist()) == ["meter-a", "meter-a", "meter-b"]


def test_validation_split_never_separates_crops_from_one_image():
    groups = np.array(["source-a"] * 4 + ["source-b"] * 3 + ["source-c"] * 2 + ["source-d"])

    train, validation = split_by_group(groups, np.random.default_rng(4))

    assert set(groups[train]).isdisjoint(set(groups[validation]))
    assert len(train) + len(validation) == len(groups)


def test_trained_ensemble_reads_all_confirmed_samples():
    pytest.importorskip("torch")
    ensemble = ConfidenceEnsemble(
        load_classifier("models/digit_cnn_samples_ft.pt"),
        load_classifier("models/digit_cnn_samples.pt"),
    )
    expected = [2927.4, 4859.12, 4869.4, 6308.0, 2878.5]

    actual = [
        read_meter(f"samples/meter_{index:02}.png", clf=ensemble)["detected_reading"]
        for index in range(1, 6)
    ]

    assert actual == expected
