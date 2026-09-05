# hog.py modified file
"""Task 1: HOG feature extraction, recognition-rate evaluation and visualisation.

The 1-Nearest Neighbour model is used only as a fixed baseline evaluator.
The LBP experiment must use the same data split and evaluator settings so
that the two feature extraction techniques can be compared fairly.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TRAIN_DIR = Path(r"C:\Y3S1\mini project\mini project 2 code\Training")

ANNOTATION_FILE = TRAIN_DIR / "annotations.csv"
TEST_DIR = TRAIN_DIR.parent / "Testing Images"

IMAGE_SIZE = (64, 64)
DISPLAY_COUNT = 10
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42


# ---------------------------------------------------------
# Read and clean annotations
# ---------------------------------------------------------


def read_annotations():
    labels_by_filename = {}

    with ANNOTATION_FILE.open(newline="", encoding="utf-8-sig") as file:

        reader = csv.DictReader(file)

        for row in reader:
            filename = row["file_name"].strip()
            category = int(float(row["category"]))
            image_path = TRAIN_DIR / filename

            if not image_path.exists():
                continue

            # Keep only one annotation for each filename
            labels_by_filename.setdefault(filename, category)

    return labels_by_filename


def image_hash(path):
    """Return a SHA-256 hash used to detect exact train-test duplicates."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_colour_mask(image):
    """Create a mask for common red, blue and yellow traffic-sign colours."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 40), (10, 255, 255)),
        cv2.inRange(hsv, (170, 70, 40), (180, 255, 255)),
    )
    blue = cv2.inRange(hsv, (90, 60, 35), (140, 255, 255))
    yellow = cv2.inRange(hsv, (15, 70, 45), (40, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red, blue), yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def extract_sign_region(image):
    """Automatically locate and crop the most likely traffic-sign region."""
    mask = make_colour_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
# Task 1: HOG feature extraction
# ---------------------------------------------------------


def extract_hog(image, create_visualisation=False):
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Standardise image size
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    result = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=create_visualisation,
        feature_vector=True,
    )

    if create_visualisation:
        features, hog_image = result

        hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        return features.astype(np.float32), gray, hog_image

    return result.astype(np.float32)


# ---------------------------------------------------------
# Prepare popup visualisation
# ---------------------------------------------------------


def add_title(image, title):
    panel = cv2.resize(image, (300, 300))

    title_area = np.zeros((45, panel.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        title_area,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.vconcat([title_area, panel])


def create_display(image, cropped_sign, gray, hog_image, filename, expected, predicted):
    original_panel = add_title(image, "Original")

    cropped_panel = add_title(cropped_sign, "Cropped Sign")

    grayscale_panel = add_title(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale")

    hog_panel = add_title(cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR), "HOG Features")

    panels = cv2.hconcat([original_panel, cropped_panel, grayscale_panel, hog_panel])

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
        cv2.LINE_AA,
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
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        "Any key = next image | Esc = stop",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    return cv2.vconcat([header, panels])


def terminal_status(correct):
    """Return green/red status text when the terminal supports ANSI colour."""
    text = "CORRECT" if correct else "WRONG"
    if not sys.stdout.isatty():
        return text
    colour = "\033[92m" if correct else "\033[91m"
    return f"{colour}{text}\033[0m"


# ---------------------------------------------------------
# Run Task 1 HOG recognition-rate experiment
# ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Print results without opening the visualisation window.",
    )
    options = parser.parse_args()

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"annotations.csv was not found: {ANNOTATION_FILE}")

    labels_by_filename = read_annotations()
    test_paths = sorted(
        path for path in TEST_DIR.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_paths:
        raise FileNotFoundError(f"No testing images were found: {TEST_DIR}")

    # The lecturer test set must remain independent from model fitting.
    test_hashes = {image_hash(path) for path in test_paths}
    training_images = sorted(
        [
            (TRAIN_DIR / filename, category)
            for filename, category in labels_by_filename.items()
            if image_hash(TRAIN_DIR / filename) not in test_hashes
        ],
        key=lambda item: item[0].name.lower(),
    )
    removed_count = len(labels_by_filename) - len(training_images)

    feature_list = []
    label_list = []
    valid_training_paths = []

    print("=" * 68)
    print("TASK 1: HOG FEATURE EXTRACTION AND RECOGNITION RATE")
    print("=" * 68)
    print(f"Unique training images           : {len(labels_by_filename)}")
    print(f"Exact test-image matches removed : {removed_count}")
    print(f"Clean training images            : {len(training_images)}")
    print(f"Lecturer testing images          : {len(test_paths)}")
    print("Extracting HOG features...")

    for image_path, category in training_images:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        cropped_sign = extract_sign_region(image)
        features = extract_hog(cropped_sign)

        feature_list.append(features)
        label_list.append(category)

        # IMPORTANT:
        # Save the path in exactly the same order as X_all
        valid_training_paths.append(str(image_path.resolve()))

    if not feature_list:
        raise RuntimeError("No valid training images could be processed.")

    # Complete HOG feature matrix before the stratified 80/20 split.
    X_all = np.vstack(feature_list)
    y_all = np.asarray(label_list)

    print("\nHOG extraction completed.")
    print(f"Images successfully processed   : {len(X_all)}")
    print(f"HOG features per image          : {X_all.shape[1]}")
    print(f"HOG feature matrix shape        : {X_all.shape}")
    print(f"Label array shape               : {y_all.shape}")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_all,
        y_all,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )

    print("\nFixed evaluation protocol")
    print("Evaluator                        : 1-Nearest Neighbour")
    print("Distance                         : Euclidean")
    print("Feature normalisation            : L2")
    print(f"Training samples                 : {len(X_train)}")
    print(f"Validation samples               : {len(X_validation)}")

    evaluator = make_pipeline(
        Normalizer(norm="l2"),
        KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1),
    )

    # Validation recognition rate: fit only on the 80% training partition.
    evaluator.fit(X_train, y_train)
    validation_predictions = evaluator.predict(X_validation)
    validation_correct = int(np.sum(validation_predictions == y_validation))
    validation_total = len(y_validation)
    validation_rate = validation_correct / validation_total

    # Apply exactly the same HOG preprocessing to the lecturer test images.
    test_features = []
    test_labels = []
    valid_test_paths = []

    for image_path in test_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable testing image: {image_path.name}")
            continue

        cropped_sign = extract_sign_region(image)
        test_features.append(extract_hog(cropped_sign))
        test_labels.append(int(image_path.name.split("_")[0]))
        valid_test_paths.append(image_path)

    if not test_features:
        raise RuntimeError("No valid lecturer testing images could be processed.")

    X_test = np.vstack(test_features)
    y_test = np.asarray(test_labels)

    # Save extracted HOG datasets for Task 2
    BASE_DIR = Path(__file__).parent

    np.save(BASE_DIR / "hog_X_all.npy", X_all)
    np.save(BASE_DIR / "hog_y_all.npy", y_all)

    np.save(BASE_DIR / "hog_X_test.npy", X_test)
    np.save(BASE_DIR / "hog_y_test.npy", y_test)

    # Save image paths in EXACTLY the same order as feature rows
    np.save(
        BASE_DIR / "hog_train_paths.npy", np.array(valid_training_paths, dtype=object)
    )

    np.save(
        BASE_DIR / "hog_test_paths.npy",
        np.array([str(path.resolve()) for path in valid_test_paths], dtype=object),
    )

    print("\nHOG datasets saved for Task 2:")
    print(f"hog_X_all.npy          : {X_all.shape}")
    print(f"hog_y_all.npy          : {y_all.shape}")
    print(f"hog_X_test.npy         : {X_test.shape}")
    print(f"hog_y_test.npy         : {y_test.shape}")
    print(f"hog_train_paths.npy    : {len(valid_training_paths)} paths")
    print(f"hog_test_paths.npy     : {len(valid_test_paths)} paths")

    # After validation, refit the same evaluator on all clean training data.
    evaluator.fit(X_all, y_all)
    test_predictions = evaluator.predict(X_test)
    test_correct = int(np.sum(test_predictions == y_test))
    test_total = len(y_test)
    test_rate = test_correct / test_total

    print("\n" + "=" * 68)
    print("LECTURER TEST PREDICTIONS")
    print("=" * 68)
    for number, (path, expected, predicted) in enumerate(
        zip(valid_test_paths, y_test, test_predictions), start=1
    ):
        correct = expected == predicted
        print(
            f"{number:>3}. {path.name:<22} "
            f"expected={expected:03d} predicted={predicted:03d} "
            f"{terminal_status(correct)}"
        )

    print("\n" + "=" * 68)
    print("HOG RECOGNITION-RATE RESULTS")
    print("=" * 68)
    print(
        f"Validation recognition rate     : {validation_correct}/"
        f"{validation_total} = {validation_rate * 100:.2f}%"
    )
    print(
        f"Lecturer test recognition rate  : {test_correct}/"
        f"{test_total} = {test_rate * 100:.2f}%"
    )
    print("Fixed evaluator                  : 1-Nearest Neighbour")
    print(f"Visualisation results            : {min(DISPLAY_COUNT, test_total)}")

    if options.no_popup:
        return

    print("\nPress any key for the next image.")
    print("Press Esc to stop.")

    cv2.namedWindow("Task 1 - HOG Feature Extraction", cv2.WINDOW_NORMAL)

    indices = np.linspace(
        0, test_total - 1, num=min(DISPLAY_COUNT, test_total), dtype=int
    )

    for index in indices:
        image_path = valid_test_paths[index]
        image = cv2.imread(str(image_path))
        cropped_sign = extract_sign_region(image)
        _, gray, hog_image = extract_hog(cropped_sign, create_visualisation=True)
        display = create_display(
            image,
            cropped_sign,
            gray,
            hog_image,
            image_path.name,
            int(y_test[index]),
            int(test_predictions[index]),
        )

        cv2.imshow("Task 1 - HOG Feature Extraction", display)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Task 2: Random Forest Classification using HOG Features"""

import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Shared preprocessing function from Task 1
from hog import extract_sign_region

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent

# Task 1 generated datasets
HOG_X_ALL = BASE_DIR / "hog_X_all.npy"
HOG_Y_ALL = BASE_DIR / "hog_y_all.npy"

HOG_X_TEST = BASE_DIR / "hog_X_test.npy"
HOG_Y_TEST = BASE_DIR / "hog_y_test.npy"

# Image paths saved by Task 1
TRAIN_PATHS_FILE = BASE_DIR / "hog_train_paths.npy"
TEST_PATHS_FILE = BASE_DIR / "hog_test_paths.npy"

RANDOM_SEED = 42

NUM_VISUAL_TRAINING = 10
NUM_VISUAL_TEST = 10


# ============================================================
# LOAD HOG DATASET
# ============================================================


def load_hog_dataset():

    print("\nLoading HOG features generated from Task 1...")

    required_files = [
        HOG_X_ALL,
        HOG_Y_ALL,
        HOG_X_TEST,
        HOG_Y_TEST,
        TRAIN_PATHS_FILE,
        TEST_PATHS_FILE,
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"\nRequired Task 1 file not found:\n{file}\n\n"
                "Please run hog.py first to generate the HOG datasets."
            )

    X_all = np.load(HOG_X_ALL)
    y_all = np.load(HOG_Y_ALL)

    X_test = np.load(HOG_X_TEST)
    y_test = np.load(HOG_Y_TEST)

    training_paths = [
        Path(path) for path in np.load(TRAIN_PATHS_FILE, allow_pickle=True)
    ]

    test_paths = [Path(path) for path in np.load(TEST_PATHS_FILE, allow_pickle=True)]

    print("\nHOG dataset loaded successfully.")

    print(f"Training feature matrix : {X_all.shape}")
    print(f"Training labels         : {y_all.shape}")

    print(f"Test feature matrix     : {X_test.shape}")
    print(f"Test labels             : {y_test.shape}")

    print(f"Training image paths    : {len(training_paths)}")
    print(f"Test image paths        : {len(test_paths)}")

    return (
        X_all,
        y_all,
        X_test,
        y_test,
        training_paths,
        test_paths,
    )


# ============================================================
# CALCULATE METRICS
# ============================================================


def calculate_metrics(y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return accuracy, precision, recall, f1


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================


def create_random_forest():

    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ============================================================
# TRAIN / VALIDATION EXPERIMENT
# ============================================================


def train_and_validate(
    X_all,
    y_all,
    training_paths,
):

    print("\n" + "=" * 65)
    print("TRAIN / VALIDATION EXPERIMENT")
    print("=" * 65)

    indices = np.arange(len(X_all))

    (
        train_indices,
        validation_indices,
    ) = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )

    X_train = X_all[train_indices]
    X_validation = X_all[validation_indices]

    y_train = y_all[train_indices]
    y_validation = y_all[validation_indices]

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_validation)}")

    model = create_random_forest()

    print("\nTraining Random Forest...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"Training completed in " f"{training_time:.2f} seconds.")

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions = model.predict(X_validation)

    accuracy, precision, recall, f1 = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    print(f"Accuracy  : {accuracy:.4f} " f"({accuracy * 100:.2f}%)")

    print(f"Precision : {precision:.4f} " f"({precision * 100:.2f}%)")

    print(f"Recall    : {recall:.4f} " f"({recall * 100:.2f}%)")

    print(f"F1-score  : {f1:.4f} " f"({f1 * 100:.2f}%)")

    # --------------------------------------------------------
    # Validation Confusion Matrix
    # --------------------------------------------------------

    print("\nGenerating validation confusion matrix...")

    cm = confusion_matrix(y_validation, validation_predictions)

    fig, ax = plt.subplots(figsize=(16, 14))

    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot(
        ax=ax,
        xticks_rotation="vertical",
    )

    ax.set_title("HOG + Random Forest - Validation Confusion Matrix")

    plt.tight_layout()

    plt.savefig(BASE_DIR / "random_forest_validation_confusion_matrix.png", dpi=150)

    plt.close()

    print("Saved: " "random_forest_validation_confusion_matrix.png")

    return (
        model,
        train_indices,
        validation_indices,
    )


# ============================================================
# CREATE RESULT DISPLAY
# ============================================================


def add_title(image, title):

    panel = cv2.resize(image, (300, 300))

    title_area = np.zeros((45, 300, 3), dtype=np.uint8)

    cv2.putText(
        title_area,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.vconcat([title_area, panel])


def create_prediction_display(
    original,
    cropped_sign,
    filename,
    actual,
    predicted,
    dataset_type,
):

    original_panel = add_title(original, "Original Image")

    cropped_panel = add_title(cropped_sign, "Detected Sign")

    result_panel = np.zeros((300, 300, 3), dtype=np.uint8)

    correct = actual == predicted

    status = "CORRECT" if correct else "WRONG"

    status_colour = (0, 255, 0) if correct else (0, 0, 255)

    cv2.putText(
        result_panel,
        f"Actual: {actual:03d}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        result_panel,
        f"Predicted: {predicted:03d}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        result_panel,
        status,
        (50, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_colour,
        2,
        cv2.LINE_AA,
    )

    result_panel = add_title(result_panel, "Random Forest Result")

    panels = cv2.hconcat(
        [
            original_panel,
            cropped_panel,
            result_panel,
        ]
    )

    header = np.zeros((70, panels.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        header,
        f"{dataset_type} | File: {filename}",
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.vconcat([header, panels])


# ============================================================
# VISUALISE RESULTS
# ============================================================


def visualise_results(
    image_paths,
    labels,
    predictions,
    indices,
    dataset_type,
):

    print("\n" + "=" * 65)
    print(f"VISUALISING {len(indices)} " f"{dataset_type.upper()} RESULTS")
    print("=" * 65)

    print("Press any key for next image.")
    print("Press Esc to stop.")

    window_name = f"Task 2 - Random Forest - {dataset_type}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for number, index in enumerate(indices, start=1):

        image_path = image_paths[index]

        image = cv2.imread(str(image_path))

        if image is None:

            print(f"Could not load image: " f"{image_path}")

            continue

        cropped_sign = extract_sign_region(image)

        actual = int(labels[index])
        predicted = int(predictions[index])

        display = create_prediction_display(
            original=image,
            cropped_sign=cropped_sign,
            filename=image_path.name,
            actual=actual,
            predicted=predicted,
            dataset_type=(f"{dataset_type} " f"{number}/{len(indices)}"),
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:

            print("Visualisation stopped by user.")

            break

    cv2.destroyWindow(window_name)


# ============================================================
# VISUALISE 10 TRAINING RESULTS
# ============================================================


def visualise_training_results(
    model,
    X_all,
    y_all,
    training_paths,
    train_indices,
):

    print("\nPreparing training result visualisation...")

    # Predict all images using the validation experiment model
    all_predictions = model.predict(X_all)

    # Select only images from the actual 80% training partition
    rng = np.random.default_rng(RANDOM_SEED)

    selected_indices = rng.choice(
        train_indices,
        size=min(NUM_VISUAL_TRAINING, len(train_indices)),
        replace=False,
    )

    print(f"Selected {len(selected_indices)} " "training images.")

    visualise_results(
        image_paths=training_paths,
        labels=y_all,
        predictions=all_predictions,
        indices=selected_indices,
        dataset_type="Training",
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================


def train_final_model(
    X_all,
    y_all,
):

    print("\n" + "=" * 65)
    print("FINAL RANDOM FOREST TRAINING")
    print("=" * 65)

    model = create_random_forest()

    print(f"Training final model using " f"all {len(X_all)} training images...")

    start_time = time.time()

    model.fit(X_all, y_all)

    training_time = time.time() - start_time

    print(f"Final training completed in " f"{training_time:.2f} seconds.")

    return model


# ============================================================
# FINAL TESTING
# ============================================================


def test_random_forest(
    model,
    X_test,
    y_test,
):

    print("\n" + "=" * 65)
    print("FINAL TESTING - 84 TEST IMAGES")
    print("=" * 65)

    print("\nMaking predictions...")

    start_time = time.time()

    predictions = model.predict(X_test)

    prediction_time = time.time() - start_time

    accuracy, precision, recall, f1 = calculate_metrics(y_test, predictions)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(f"Accuracy  : {accuracy:.4f} " f"({accuracy * 100:.2f}%)")

    print(f"Precision : {precision:.4f} " f"({precision * 100:.2f}%)")

    print(f"Recall    : {recall:.4f} " f"({recall * 100:.2f}%)")

    print(f"F1-score  : {f1:.4f} " f"({f1 * 100:.2f}%)")

    correct_predictions = np.sum(predictions == y_test)

    print(f"\nCorrect predictions: " f"{correct_predictions}/{len(y_test)}")

    print(f"Prediction time: " f"{prediction_time:.4f} seconds")

    # --------------------------------------------------------
    # Test confusion matrix
    # --------------------------------------------------------

    print("\nGenerating final test confusion matrix...")

    cm_test = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(16, 14))

    display = ConfusionMatrixDisplay(confusion_matrix=cm_test)

    display.plot(
        ax=ax,
        xticks_rotation="vertical",
    )

    ax.set_title("HOG + Random Forest - Final Test Confusion Matrix")

    plt.tight_layout()

    plt.savefig(BASE_DIR / "random_forest_test_confusion_matrix.png", dpi=150)

    plt.close()

    print("Saved: " "random_forest_test_confusion_matrix.png")

    return predictions


# ============================================================
# VISUALISE 10 TEST RESULTS
# ============================================================


def visualise_selected_test_results(
    test_paths,
    y_test,
    predictions,
):

    # Use evenly distributed images instead of just first 10
    selected_indices = np.linspace(
        0,
        len(y_test) - 1,
        num=min(NUM_VISUAL_TEST, len(y_test)),
        dtype=int,
    )

    visualise_results(
        image_paths=test_paths,
        labels=y_test,
        predictions=predictions,
        indices=selected_indices,
        dataset_type="Test",
    )

    return selected_indices


# ============================================================
# REMAINING TEST RESULTS FOR APPENDIX
# ============================================================


def visualise_remaining_test_results(
    test_paths,
    y_test,
    predictions,
    selected_indices,
):

    selected_set = set(selected_indices.tolist())

    remaining_indices = np.array(
        [index for index in range(len(y_test)) if index not in selected_set]
    )

    print(f"\nRemaining test images " f"for Appendix: {len(remaining_indices)}")

    visualise_results(
        image_paths=test_paths,
        labels=y_test,
        predictions=predictions,
        indices=remaining_indices,
        dataset_type="Appendix Test",
    )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_final_summary(
    final_model,
    X_all,
    y_all,
    X_test,
    y_test,
    test_predictions,
):

    train_predictions = final_model.predict(X_all)

    (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
    ) = calculate_metrics(y_all, train_predictions)

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
    ) = calculate_metrics(y_test, test_predictions)

    print("\n" + "=" * 75)
    print("FINAL PERFORMANCE COMPARISON " "- HOG + RANDOM FOREST")
    print("=" * 75)

    print(
        f"{'Dataset':<15}"
        f"{'Accuracy':<15}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1-score':<15}"
    )

    print("-" * 75)

    print(
        f"{'Training':<15}"
        f"{train_accuracy * 100:<15.2f}"
        f"{train_precision * 100:<15.2f}"
        f"{train_recall * 100:<15.2f}"
        f"{train_f1 * 100:<15.2f}"
    )

    print(
        f"{'Test':<15}"
        f"{test_accuracy * 100:<15.2f}"
        f"{test_precision * 100:<15.2f}"
        f"{test_recall * 100:<15.2f}"
        f"{test_f1 * 100:<15.2f}"
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("TASK 2: RANDOM FOREST " "CLASSIFICATION USING HOG FEATURES")
    print("=" * 70)

    print("\nExperimental Design:")
    print("HOG Feature Extraction " "+ Random Forest Classifier")

    # ========================================================
    # STEP 1: LOAD TASK 1 DATASET
    # ========================================================

    (
        X_all,
        y_all,
        X_test,
        y_test,
        training_paths,
        test_paths,
    ) = load_hog_dataset()

    # Safety checks
    if len(training_paths) != len(X_all):

        raise ValueError(
            f"Training paths ({len(training_paths)}) "
            f"do not match HOG features ({len(X_all)})."
        )

    if len(test_paths) != len(X_test):

        raise ValueError(
            f"Test paths ({len(test_paths)}) "
            f"do not match HOG features ({len(X_test)})."
        )

    # ========================================================
    # STEP 2: TRAIN / VALIDATION EXPERIMENT
    # ========================================================

    (
        validation_model,
        train_indices,
        validation_indices,
    ) = train_and_validate(
        X_all,
        y_all,
        training_paths,
    )

    # ========================================================
    # STEP 3: VISUALISE 10 TRAINING RESULTS
    # ========================================================

    print("\n" + "#" * 65)
    print("STEP 1: VISUALISE 10 TRAINING RESULTS")
    print("#" * 65)

    visualise_training_results(
        validation_model,
        X_all,
        y_all,
        training_paths,
        train_indices,
    )

    # ========================================================
    # STEP 4: TRAIN FINAL MODEL USING ALL DATA
    # ========================================================

    final_model = train_final_model(
        X_all,
        y_all,
    )

    # ========================================================
    # STEP 5: FINAL TESTING ON 84 TEST IMAGES
    # ========================================================

    test_predictions = test_random_forest(
        final_model,
        X_test,
        y_test,
    )

    # ========================================================
    # STEP 6: VISUALISE 10 TEST RESULTS
    # ========================================================

    print("\n" + "#" * 65)
    print("STEP 2: VISUALISE 10 TEST RESULTS")
    print("#" * 65)

    selected_test_indices = visualise_selected_test_results(
        test_paths,
        y_test,
        test_predictions,
    )

    # ========================================================
    # STEP 7: REMAINING TEST RESULTS FOR APPENDIX
    # ========================================================

    print("\n" + "#" * 65)
    print("STEP 3: REMAINING TEST RESULTS FOR APPENDIX")
    print("#" * 65)

    visualise_remaining_test_results(
        test_paths,
        y_test,
        test_predictions,
        selected_test_indices,
    )

    # ========================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================

    print_final_summary(
        final_model,
        X_all,
        y_all,
        X_test,
        y_test,
        test_predictions,
    )


if __name__ == "__main__":
    main()
