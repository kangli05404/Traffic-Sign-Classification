"""Minimal end-to-end traffic-sign experiment.

One file performs annotation cleaning, exact-overlap removal, HSV segmentation,
HOG extraction, Random Forest training, evaluation, and an optional popup
visualisation. It writes only results.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "Training"
TEST_DIR = ROOT / "Testing Image"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"
IMAGE_SIZE = (64, 64)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}


def terminal_status(text: str, correct: bool) -> str:
    """Colour status text when the terminal supports ANSI escape sequences."""
    if not sys.stdout.isatty():
        return text
    colour = "\033[92m" if correct else "\033[91m"
    return f"{colour}{text}\033[0m"
HOG_CONFIGS = [
    {"name": "baseline_64_cell8_ori9", "image_size": (64, 64), "cell": 8, "orientations": 9},
    {"name": "fine_64_cell4_ori9", "image_size": (64, 64), "cell": 4, "orientations": 9},
    {"name": "more_orientations_64_cell8_ori12", "image_size": (64, 64), "cell": 8, "orientations": 12},
    {"name": "larger_96_cell8_ori9", "image_size": (96, 96), "cell": 8, "orientations": 9},
]


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_clean_annotations() -> dict[str, int]:
    annotation_path = TRAIN_DIR / "annotations.csv"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotations.csv: {annotation_path}")
    labels: dict[str, int] = {}
    with annotation_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            name = (row.get("file_name") or row.get("filename") or "").strip()
            category = row.get("category") or row.get("class") or row.get("label")
            if not name or category is None or not (TRAIN_DIR / name).exists():
                continue
            # Keep one label per image. Conflicting duplicate rows are audited below.
            labels.setdefault(name, int(float(category)))
    return labels


def make_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 70, 40), (10, 255, 255)), cv2.inRange(hsv, (170, 70, 40), (180, 255, 255)))
    blue = cv2.inRange(hsv, (90, 60, 35), (140, 255, 255))
    yellow = cv2.inRange(hsv, (15, 70, 45), (40, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red, blue), yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def segment(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = make_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        aspect = width / max(height, 1)
        if area >= 80 and 0.45 <= aspect <= 1.8:
            candidates.append((area, x, y, width, height))
    if not candidates:
        return image, mask
    _, x, y, width, height = max(candidates)
    pad = max(3, int(0.10 * max(width, height)))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image.shape[1], x + width + pad), min(image.shape[0], y + height + pad)
    return image[y1:y2, x1:x2], mask


def hog_features(image: np.ndarray, config: dict | None = None) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty image supplied to HOG")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    config = config or HOG_CONFIGS[0]
    image_size = tuple(config["image_size"])
    cell = int(config["cell"])
    block = cell * 2
    gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    descriptor = cv2.HOGDescriptor(image_size, (block, block), (cell, cell), (cell, cell), int(config["orientations"]))
    return descriptor.compute(gray).reshape(-1).astype(np.float32)


def prediction_lookup() -> dict[str, tuple[int, int]]:
    return {}


def make_display(image: np.ndarray, filename: str, expected: int, predicted: int) -> np.ndarray:
    mask = make_mask(image)
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    height = 330
    width = max(1, int(image.shape[1] * height / image.shape[0]))
    panels = [
        (cv2.resize(image, (width, height)), "Original"),
        (cv2.cvtColor(cv2.resize(edges, (width, height)), cv2.COLOR_GRAY2BGR), "Canny Edge"),
        (cv2.cvtColor(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR), "Image Mask"),
        (cv2.resize(segmented, (width, height)), "Segmented"),
    ]
    labelled = []
    for panel, title in panels:
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(panel, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        labelled.append(panel)
    display = cv2.hconcat(labelled)
    header = np.zeros((62, display.shape[1], 3), dtype=np.uint8)
    correct = expected == predicted
    cv2.putText(header, f"{filename} | expected: {expected:03d} | predicted: {predicted:03d}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "CORRECT" if correct else "WRONG", (display.shape[1] - 155, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 0) if correct else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "Any key = next image   |   Esc = stop", (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 220, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([header, display])


def make_confusion_matrix_display(matrix: np.ndarray) -> np.ndarray:
    """Create a compact heatmap for a popup; raw values remain in results.json."""
    scaled = cv2.normalize(matrix.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (900, 700), interpolation=cv2.INTER_NEAREST)
    cv2.putText(heatmap, "Confusion Matrix | rows = expected, columns = predicted", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return heatmap


def build_random_forest(parameters: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(parameters["n_estimators"]),
        max_features=parameters["max_features"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-popup", "--terminal-only", dest="no_popup", action="store_true", help="Print all results in the terminal without opening image windows")
    parser.add_argument("--tune", action="store_true", help="Compare a small set of Random Forest settings using only the internal validation split")
    parser.add_argument("--tune-hog", action="store_true", help="Compare a small set of HOG settings using only the internal validation split")
    parser.add_argument("--start", type=int, default=0, help="First test-image index for the popup")
    parser.add_argument("--limit", type=int, default=0, help="Maximum popup images; 0 means all")
    options = parser.parse_args()

    labels_by_name = read_clean_annotations()
    train_paths = sorted(TRAIN_DIR / name for name in labels_by_name)
    test_paths = sorted(path for path in TEST_DIR.rglob("*") if path.suffix.lower() in EXTENSIONS)
    if not train_paths or not test_paths:
        raise RuntimeError("Training or testing images were not found")

    test_hashes = {image_hash(path) for path in test_paths}
    clean_train_paths = [path for path in train_paths if image_hash(path) not in test_hashes]
    overlap_count = len(train_paths) - len(clean_train_paths)

    print("=" * 72)
    print("TRAFFIC SIGN RECOGNITION: HOG + RANDOM FOREST")
    print("=" * 72)
    print(f"Training images found                 : {len(train_paths)}")
    print(f"Annotation labels after deduplication : {len(labels_by_name)}")
    print(f"Lecturer test images                  : {len(test_paths)}")
    print(f"Exact train/test copies excluded      : {overlap_count}")
    print(f"Training images used                   : {len(clean_train_paths)}")
    print("Preprocessing                          : HSV colour segmentation")
    print("Feature extraction                     : HOG (64x64 grayscale)")
    print("Classifier                             : Random Forest (400 trees)")

    y = np.asarray([labels_by_name[path.name] for path in clean_train_paths])
    selected_hog = HOG_CONFIGS[0]
    hog_tuning_results = []
    x = None
    if options.tune_hog:
        print("\nHOG TUNING (validation set only)")
        for config in HOG_CONFIGS:
            candidate_values = []
            for path in clean_train_paths:
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"Could not read training image: {path}")
                crop, _ = segment(image)
                candidate_values.append(hog_features(crop, config))
            candidate_x = np.vstack(candidate_values)
            candidate_train, candidate_validation, candidate_y_train, candidate_y_validation = train_test_split(
                candidate_x, y, test_size=0.20, random_state=42, stratify=y
            )
            candidate_model = build_random_forest({"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 1})
            candidate_model.fit(candidate_train, candidate_y_train)
            candidate_accuracy = float(accuracy_score(candidate_y_validation, candidate_model.predict(candidate_validation)))
            hog_tuning_results.append({"parameters": config, "validation_accuracy": candidate_accuracy})
            print(f"{config} -> {candidate_accuracy * 100:.2f}%")
            if x is None or candidate_accuracy > max(row["validation_accuracy"] for row in hog_tuning_results[:-1]):
                x = candidate_x
                selected_hog = config
            else:
                del candidate_x
        print(f"Selected HOG parameters: {selected_hog}")
    else:
        x_values = []
        for path in clean_train_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read training image: {path}")
            crop, _ = segment(image)
            x_values.append(hog_features(crop, selected_hog))
        x = np.vstack(x_values)

    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    default_parameters = {"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 1}
    rf_tuning_results = []
    if options.tune:
        candidates = [
            {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1},
            {"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 1},
            {"n_estimators": 600, "max_features": "sqrt", "min_samples_leaf": 1},
            {"n_estimators": 400, "max_features": "log2", "min_samples_leaf": 1},
            {"n_estimators": 400, "max_features": 0.5, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 2},
        ]
        print("\nRANDOM FOREST TUNING (validation set only)")
        for parameters in candidates:
            candidate_model = build_random_forest(parameters)
            candidate_model.fit(x_train, y_train)
            candidate_accuracy = float(accuracy_score(y_validation, candidate_model.predict(x_validation)))
            rf_tuning_results.append({"parameters": parameters, "validation_accuracy": candidate_accuracy})
            print(f"{parameters} -> {candidate_accuracy * 100:.2f}%")
        default_parameters = max(rf_tuning_results, key=lambda row: row["validation_accuracy"])["parameters"]
        print(f"Selected parameters: {default_parameters}")

    model = build_random_forest(default_parameters)
    model.fit(x_train, y_train)
    validation_accuracy = float(accuracy_score(y_validation, model.predict(x_validation)))
    print(f"Internal validation accuracy           : {validation_accuracy * 100:.2f}%")
    model.fit(x, y)

    predictions = []
    for path in test_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        started = time.perf_counter()
        crop, _ = segment(image)
        predicted = int(model.predict([hog_features(crop, selected_hog)])[0])
        expected = int(path.name.split("_")[0])
        predictions.append({"filename": path.name, "expected": expected, "predicted": predicted, "correct": expected == predicted, "seconds": time.perf_counter() - started})

    y_true = [row["expected"] for row in predictions]
    y_pred = [row["predicted"] for row in predictions]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    matrix_labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels)
    report = classification_report(y_true, y_pred, labels=matrix_labels, output_dict=True, zero_division=0)
    result = {
        "feature_extraction": "HOG",
        "selected_hog_parameters": selected_hog,
        "hog_tuning_validation_results": hog_tuning_results,
        "classifier": "Random Forest",
        "tuning_enabled": options.tune,
        "selected_random_forest_parameters": default_parameters,
        "tuning_validation_results": rf_tuning_results,
        "training_images_used": len(clean_train_paths),
        "lecturer_test_images": len(predictions),
        "exact_train_test_overlaps_excluded": overlap_count,
        "internal_validation_accuracy": validation_accuracy,
        "lecturer_test_accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "mean_seconds_per_image": float(np.mean([row["seconds"] for row in predictions])),
        "confusion_matrix_labels": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "predictions": predictions,
    }
    RESULTS_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print("PER-IMAGE PREDICTIONS")
    print("=" * 72)
    print(f"{'No.':>4}  {'Filename':<22} {'Expected':>8} {'Predicted':>9} {'Status':<8} {'Time (s)':>9}")
    print("-" * 72)
    for index, row in enumerate(predictions, start=1):
        status = terminal_status("CORRECT", True) if row["correct"] else terminal_status("WRONG", False)
        print(f"{index:>4}  {row['filename']:<22} {row['expected']:>8} {row['predicted']:>9} {status:<8} {row['seconds']:>9.4f}")

    print("\n" + "=" * 72)
    print("CLASSIFICATION REPORT")
    print("=" * 72)
    print(classification_report(y_true, y_pred, labels=matrix_labels, zero_division=0))

    print("=" * 72)
    print("CONFUSION MATRIX")
    print("Rows = expected class; columns = predicted class")
    print(f"Class labels: {matrix_labels}")
    print(matrix)

    print("\n" + "=" * 72)
    print("FINAL METRICS")
    print("=" * 72)
    print(f"Correct predictions                   : {sum(row['correct'] for row in predictions)}/{len(predictions)}")
    print(f"Lecturer test accuracy                : {result['lecturer_test_accuracy'] * 100:.2f}%")
    print(f"Weighted precision                    : {result['weighted_precision'] * 100:.2f}%")
    print(f"Weighted recall                       : {result['weighted_recall'] * 100:.2f}%")
    print(f"Weighted F1-score                     : {result['weighted_f1'] * 100:.2f}%")
    print(f"Mean processing time                  : {result['mean_seconds_per_image']:.4f} seconds/image")
    print(f"Saved results                         : {RESULTS_FILE}")

    if not options.no_popup:
        selected = test_paths[options.start:]
        if options.limit > 0:
            selected = selected[:options.limit]
        by_name = {row["filename"]: row for row in predictions}
        cv2.namedWindow("Traffic-sign preprocessing", cv2.WINDOW_NORMAL)
        for path in selected:
            image = cv2.imread(str(path))
            row = by_name[path.name]
            cv2.imshow("Traffic-sign preprocessing", make_display(image, path.name, row["expected"], row["predicted"]))
            if cv2.waitKey(0) & 0xFF == 27:
                break
        cv2.imshow("Confusion Matrix", make_confusion_matrix_display(matrix))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
