"""Task 1: Spatial HSV colour features, recognition rate and visualisation.

The data preparation, 80/20 split and fixed 1-Nearest Neighbour evaluator are
the same as in HOG.py and LBP.py. Only the feature extraction method differs.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


# ---------------------------------------------------------
# Settings shared with HOG.py and LBP.py
# ---------------------------------------------------------

TRAIN_DIR = Path(
    r"C:\Users\teeak\Desktop\Study\Y3S1"
    r"\Mini Project (Assignment 2)\Training"
)
ANNOTATION_FILE = TRAIN_DIR / "annotations.csv"
TEST_DIR = TRAIN_DIR.parent / "Testing Image"

IMAGE_SIZE = (64, 64)
DISPLAY_COUNT = 10
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42

# Best settings selected using the validation set.
HSV_CELL_SIZE = 8
HUE_BINS = 18
SATURATION_BINS = 16
VALUE_BINS = 16


# ---------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------

def read_annotations():
    """Read class labels and remove duplicate annotation rows by filename."""
    labels_by_filename = {}
    with ANNOTATION_FILE.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            filename = row["file_name"].strip()
            category = int(float(row["category"]))

            if not (TRAIN_DIR / filename).exists():
                continue

            labels_by_filename.setdefault(filename, category)

    return labels_by_filename


def image_hash(path):
    """Return SHA-256 so exact lecturer-test copies can be excluded."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------
# Image preprocessing shared with HOG.py and LBP.py
# ---------------------------------------------------------

def make_colour_mask(image):
    """Create a mask for common red, blue and yellow sign colours."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 40), (10, 255, 255)),
        cv2.inRange(hsv, (170, 70, 40), (180, 255, 255))
    )
    blue = cv2.inRange(hsv, (90, 60, 35), (140, 255, 255))
    yellow = cv2.inRange(hsv, (15, 70, 45), (40, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red, blue), yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def extract_sign_region(image):
    """Automatically crop the most likely traffic-sign region."""
    mask = make_colour_mask(image)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / max(height, 1)

        if area >= 80 and 0.45 <= aspect_ratio <= 1.8:
            candidates.append((area, x, y, width, height))

    if not candidates:
        return image

    _, x, y, width, height = max(candidates)
    padding = max(3, int(0.10 * max(width, height)))
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)
    cropped_sign = image[y1:y2, x1:x2]

    return cropped_sign if cropped_sign.size else image


# ---------------------------------------------------------
# Task 1: Spatial HSV colour-histogram extraction
# ---------------------------------------------------------

def prepare_hsv(image):
    """Apply the shared crop and resize, then convert BGR to HSV."""
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    cropped_sign = extract_sign_region(image)
    resized = cv2.resize(
        cropped_sign,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    return cropped_sign, hsv


def normalised_histogram(values, bins, value_range):
    histogram = cv2.calcHist(
        [values],
        [0],
        None,
        [bins],
        list(value_range)
    ).flatten().astype(np.float32)
    histogram /= histogram.sum() + 1e-7
    return histogram


def extract_hsv_features(hsv):
    """Extract H, S and V histograms from an 8 x 8 spatial grid."""
    feature_vector = []

    for y_position in range(0, IMAGE_SIZE[1], HSV_CELL_SIZE):
        for x_position in range(0, IMAGE_SIZE[0], HSV_CELL_SIZE):
            cell = hsv[
                y_position:y_position + HSV_CELL_SIZE,
                x_position:x_position + HSV_CELL_SIZE
            ]
            feature_vector.extend(
                normalised_histogram(cell[:, :, 0], HUE_BINS, (0, 180))
            )
            feature_vector.extend(
                normalised_histogram(
                    cell[:, :, 1], SATURATION_BINS, (0, 256)
                )
            )
            feature_vector.extend(
                normalised_histogram(cell[:, :, 2], VALUE_BINS, (0, 256))
            )

    return np.asarray(feature_vector, dtype=np.float32)


# ---------------------------------------------------------
# Visualisation
# ---------------------------------------------------------

def add_title(image, title):
    panel = cv2.resize(image, (300, 300))
    title_area = np.zeros((45, 300, 3), dtype=np.uint8)
    cv2.putText(
        title_area,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return cv2.vconcat([title_area, panel])


def create_histogram_visualisation(hsv):
    """Draw whole-image H, S and V histograms for presentation."""
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    specifications = (
        (0, 180, (0, 0, 255), "H"),
        (1, 256, (0, 255, 0), "S"),
        (2, 256, (255, 150, 0), "V")
    )

    for channel, upper_range, colour, label in specifications:
        histogram = cv2.calcHist(
            [hsv], [channel], None, [64], [0, upper_range]
        ).flatten()
        histogram = cv2.normalize(
            histogram, None, 0, 235, cv2.NORM_MINMAX
        ).flatten()
        points = np.array([
            [int(index * 299 / 63), 270 - int(value)]
            for index, value in enumerate(histogram)
        ], dtype=np.int32)
        cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (12 + channel * 42, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA
        )

    return canvas


def create_display(image, cropped_sign, hsv, filename, expected, predicted):
    hue = cv2.normalize(
        hsv[:, :, 0], None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    histogram_image = create_histogram_visualisation(hsv)
    panels = cv2.hconcat([
        add_title(image, "Original"),
        add_title(cropped_sign, "Cropped Sign"),
        add_title(cv2.cvtColor(hue, cv2.COLOR_GRAY2BGR), "Hue Channel"),
        add_title(histogram_image, "HSV Histograms")
    ])

    header = np.zeros((70, panels.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        (
            f"File: {filename} | Expected: {expected:03d} | "
            f"Predicted: {predicted:03d}"
        ),
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    correct = expected == predicted
    cv2.putText(
        header,
        "CORRECT" if correct else "WRONG",
        (panels.shape[1] - 145, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 220, 0) if correct else (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        header,
        "Any key = next image | Esc = stop",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA
    )
    return cv2.vconcat([header, panels])


def terminal_status(correct):
    text = "CORRECT" if correct else "WRONG"
    if not sys.stdout.isatty():
        return text
    colour = "\033[92m" if correct else "\033[91m"
    return f"{colour}{text}\033[0m"


# ---------------------------------------------------------
# Run the fixed Task 1 evaluation protocol
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Print results without opening the visualisation window."
    )
    options = parser.parse_args()

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"annotations.csv was not found: {ANNOTATION_FILE}"
        )

    labels_by_filename = read_annotations()
    test_paths = sorted(
        path for path in TEST_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_paths:
        raise FileNotFoundError(f"No testing images were found: {TEST_DIR}")

    test_hashes = {image_hash(path) for path in test_paths}
    training_images = sorted([
        (TRAIN_DIR / filename, category)
        for filename, category in labels_by_filename.items()
        if image_hash(TRAIN_DIR / filename) not in test_hashes
    ], key=lambda item: item[0].name.lower())
    removed_count = len(labels_by_filename) - len(training_images)

    feature_list = []
    label_list = []

    print("=" * 68)
    print("TASK 1: SPATIAL HSV FEATURES AND RECOGNITION RATE")
    print("=" * 68)
    print(f"Unique training images           : {len(labels_by_filename)}")
    print(f"Exact test-image matches removed : {removed_count}")
    print(f"Clean training images            : {len(training_images)}")
    print(f"Lecturer testing images          : {len(test_paths)}")
    print("HSV spatial grid                 : 8 x 8")
    print("Histogram bins                   : H=18, S=16, V=16")
    print("Extracting Spatial HSV features...")

    for image_path, category in training_images:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        _, hsv = prepare_hsv(image)
        feature_list.append(extract_hsv_features(hsv))
        label_list.append(category)

    if not feature_list:
        raise RuntimeError("No valid training images could be processed.")

    X_all = np.vstack(feature_list)
    y_all = np.asarray(label_list)

    print("\nSpatial HSV extraction completed.")
    print(f"Images successfully processed   : {len(X_all)}")
    print(f"HSV features per image          : {X_all.shape[1]}")
    print(f"HSV feature matrix shape        : {X_all.shape}")
    print(f"Label array shape               : {y_all.shape}")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_all,
        y_all,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_all
    )

    print("\nFixed evaluation protocol")
    print("Evaluator                        : 1-Nearest Neighbour")
    print("Distance                         : Euclidean")
    print("Feature normalisation            : L2")
    print(f"Training samples                 : {len(X_train)}")
    print(f"Validation samples               : {len(X_validation)}")

    evaluator = make_pipeline(
        Normalizer(norm="l2"),
        KNeighborsClassifier(
            n_neighbors=1,
            metric="euclidean",
            n_jobs=-1
        )
    )
    evaluator.fit(X_train, y_train)
    validation_predictions = evaluator.predict(X_validation)
    validation_correct = int(np.sum(validation_predictions == y_validation))
    validation_total = len(y_validation)
    validation_rate = validation_correct / validation_total

    test_features = []
    test_labels = []
    valid_test_paths = []

    for image_path in test_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable testing image: {image_path.name}")
            continue

        _, hsv = prepare_hsv(image)
        test_features.append(extract_hsv_features(hsv))
        test_labels.append(int(image_path.name.split("_")[0]))
        valid_test_paths.append(image_path)

    if not test_features:
        raise RuntimeError("No valid lecturer testing images could be processed.")

    X_test = np.vstack(test_features)
    y_test = np.asarray(test_labels)

    evaluator.fit(X_all, y_all)
    test_predictions = evaluator.predict(X_test)
    test_correct = int(np.sum(test_predictions == y_test))
    test_total = len(y_test)
    test_rate = test_correct / test_total

    print("\n" + "=" * 68)
    print("LECTURER TEST PREDICTIONS")
    print("=" * 68)
    for number, (path, expected, predicted) in enumerate(
        zip(valid_test_paths, y_test, test_predictions),
        start=1
    ):
        correct = expected == predicted
        print(
            f"{number:>3}. {path.name:<22} "
            f"expected={expected:03d} predicted={predicted:03d} "
            f"{terminal_status(correct)}"
        )

    print("\n" + "=" * 68)
    print("SPATIAL HSV RECOGNITION-RATE RESULTS")
    print("=" * 68)
    print(
        f"Validation recognition rate     : {validation_correct}/"
        f"{validation_total} = {validation_rate * 100:.2f}%"
    )
    print(
        f"Lecturer test recognition rate  : {test_correct}/"
        f"{test_total} = {test_rate * 100:.2f}%"
    )
    print("HSV configuration                : C8-H18-S16-V16")
    print("Fixed evaluator                  : 1-Nearest Neighbour")
    print(f"Visualisation results            : {min(DISPLAY_COUNT, test_total)}")

    if options.no_popup:
        return

    print("\nPress any key for the next image.")
    print("Press Esc to stop.")
    cv2.namedWindow("Task 1 - Spatial HSV Colour Features", cv2.WINDOW_NORMAL)
    indices = np.linspace(
        0,
        test_total - 1,
        num=min(DISPLAY_COUNT, test_total),
        dtype=int
    )

    for index in indices:
        image_path = valid_test_paths[index]
        image = cv2.imread(str(image_path))
        cropped_sign, hsv = prepare_hsv(image)
        display = create_display(
            image,
            cropped_sign,
            hsv,
            image_path.name,
            int(y_test[index]),
            int(test_predictions[index])
        )
        cv2.imshow("Task 1 - Spatial HSV Colour Features", display)

        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
