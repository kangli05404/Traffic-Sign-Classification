"""
Combined Task 1 (HOG feature extraction) + Task 2 (Random Forest classification).
"""

import csv
import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

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

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent

TRAIN_DIR = BASE_DIR / "Training"
TEST_DIR = BASE_DIR / "Testing Images"
ANNOTATION_FILE = TRAIN_DIR / "annotations.csv"

IMAGE_SIZE = (64, 64)
DISPLAY_COUNT = 20
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
RANDOM_SEED = 42

NUM_VISUAL_TRAINING = 10
NUM_VISUAL_TEST = 10

HOG_WINDOW_NAME = "Task 1 - HOG Feature Extraction"


# ============================================================
# PART 1: HOG FEATURE EXTRACTION (was hog.py)
# ============================================================


def read_annotations():
    """Read image filenames and traffic-sign categories from annotations.csv."""
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"annotations.csv was not found:\n{ANNOTATION_FILE}")

    labels_by_filename = {}
    with ANNOTATION_FILE.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row["file_name"].strip()
            category = int(float(row["category"]))
            image_path = TRAIN_DIR / filename
            if not image_path.exists():
                continue
            # Keep only one annotation per filename
            labels_by_filename.setdefault(filename, category)

    return labels_by_filename


def image_hash(path):
    """SHA-256 hash used to detect exact duplicate images between train/test."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_colour_mask(image):
    """Detect common traffic-sign colours: red, blue and yellow."""
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


def extract_hog(image, create_visualisation=False):
    """Extract HOG features from one traffic-sign image (grayscale, 64x64)."""
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    if not create_visualisation:
        return result.astype(np.float32)

    features, hog_image = result
    hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return features.astype(np.float32), gray, hog_image


def valid_hog_vector(features):
    """Check whether a HOG feature vector is a valid, finite 1-D array."""
    return features.ndim == 1 and len(features) > 0 and np.all(np.isfinite(features))


def add_title(image, title):
    """Add a title bar above an image panel. Shared by HOG and RF displays."""
    panel = cv2.resize(image, (300, 300))
    title_area = np.zeros((45, 300, 3), dtype=np.uint8)
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


def create_hog_display(image, cropped_sign, gray, hog_image, filename, category):
    """Four-panel HOG visualisation: original / crop / grayscale / HOG."""
    original_panel = add_title(image, "Original")
    cropped_panel = add_title(cropped_sign, "Extracted Sign")
    grayscale_panel = add_title(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale")
    hog_panel = add_title(cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR), "HOG Features")

    panels = cv2.hconcat([original_panel, cropped_panel, grayscale_panel, hog_panel])
    header = np.zeros((70, panels.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        header,
        f"File: {filename} | Class: {category:03d}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
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


def extract_hog_features():
    """
    Run the full Task 1 pipeline: read annotations, remove train/test
    duplicates, extract HOG features for training and lecturer test images,
    save the datasets to disk, and return everything Task 2 needs.
    """
    print("=" * 70)
    print("TASK 1: HOG FEATURE EXTRACTION")
    print("=" * 70)
    print(f"\nTraining directory:\n{TRAIN_DIR}")
    print(f"\nTesting directory:\n{TEST_DIR}")
    print(f"\nAnnotation file:\n{ANNOTATION_FILE}")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training folder was not found:\n{TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Testing folder was not found:\n{TEST_DIR}")
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"annotations.csv was not found:\n{ANNOTATION_FILE}")

    print("\nReading training annotations...")
    labels_by_filename = read_annotations()

    test_paths = sorted(
        path for path in TEST_DIR.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not test_paths:
        raise FileNotFoundError(f"No testing images found:\n{TEST_DIR}")

    # --- Remove exact train/test duplicates ---
    print("\nChecking for exact duplicates between training and test datasets...")
    test_hashes = set()
    for path in test_paths:
        try:
            test_hashes.add(image_hash(path))
        except OSError:
            print(f"Could not hash test image: {path.name}")

    training_images = []
    for filename, category in labels_by_filename.items():
        image_path = TRAIN_DIR / filename
        try:
            if image_hash(image_path) not in test_hashes:
                training_images.append((image_path, category))
        except OSError:
            print(f"Could not read training image: {image_path.name}")

    training_images = sorted(training_images, key=lambda item: item[0].name.lower())
    removed_count = len(labels_by_filename) - len(training_images)

    print(f"\nAnnotated training images : {len(labels_by_filename)}")
    print(f"Duplicates removed        : {removed_count}")
    print(f"Clean training images     : {len(training_images)}")
    print(f"Lecturer testing images   : {len(test_paths)}")
    print(f"Image size                : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")

    # --- Extract HOG from training data ---
    print("\n" + "=" * 70)
    print("EXTRACTING HOG FEATURES FROM TRAINING DATA")
    print("=" * 70)

    training_feature_list, training_label_list = [], []
    valid_training_paths, failed_training = [], []

    for index, (image_path, category) in enumerate(training_images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            failed_training.append(image_path.name)
            continue

        try:
            cropped_sign = extract_sign_region(image)
            features = extract_hog(cropped_sign)
            if not valid_hog_vector(features):
                raise ValueError("Invalid HOG vector")

            training_feature_list.append(features)
            training_label_list.append(category)
            valid_training_paths.append(str(image_path.resolve()))
        except (cv2.error, ValueError) as error:
            failed_training.append(f"{image_path.name}: {error}")

        if index % 500 == 0 or index == len(training_images):
            print(f"Processed {index}/{len(training_images)} training images")

    if not training_feature_list:
        raise RuntimeError("No valid training HOG features were extracted.")

    X_all = np.vstack(training_feature_list)
    y_all = np.asarray(training_label_list)

    print("\nTraining HOG extraction completed.")
    print(f"Successfully processed : {len(X_all)}")
    print(f"HOG features/image    : {X_all.shape[1]}")
    print(f"Feature matrix shape  : {X_all.shape}")
    print(f"Label array shape     : {y_all.shape}")

    # --- Extract HOG from test data ---
    print("\n" + "=" * 70)
    print("EXTRACTING HOG FEATURES FROM TEST DATA")
    print("=" * 70)

    test_feature_list, test_label_list = [], []
    valid_test_paths, failed_test = [], []
    display_results = []

    selected_indices = set(
        np.linspace(
            0, len(test_paths) - 1, num=min(DISPLAY_COUNT, len(test_paths)), dtype=int
        ).tolist()
    )

    for index, image_path in enumerate(test_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            failed_test.append(image_path.name)
            continue

        try:
            cropped_sign = extract_sign_region(image)

            if index in selected_indices:
                features, gray, hog_image = extract_hog(
                    cropped_sign, create_visualisation=True
                )
                category = int(image_path.name.split("_")[0])
                display_results.append(
                    (image, cropped_sign, gray, hog_image, image_path.name, category)
                )
            else:
                features = extract_hog(cropped_sign)

            if not valid_hog_vector(features):
                raise ValueError("Invalid HOG vector")

            category = int(image_path.name.split("_")[0])
            test_feature_list.append(features)
            test_label_list.append(category)
            valid_test_paths.append(str(image_path.resolve()))
        except (cv2.error, ValueError) as error:
            failed_test.append(f"{image_path.name}: {error}")

    if not test_feature_list:
        raise RuntimeError("No valid testing HOG features were extracted.")

    X_test = np.vstack(test_feature_list)
    y_test = np.asarray(test_label_list)

    print("\nTest HOG extraction completed.")
    print(f"Successfully processed : {len(X_test)}")
    print(f"HOG features/image    : {X_test.shape[1]}")
    print(f"Feature matrix shape  : {X_test.shape}")
    print(f"Label array shape     : {y_test.shape}")

    # --- Save datasets (kept so other scripts, e.g. an SVM comparison, can reuse them) ---
    print("\n" + "=" * 70)
    print("SAVING HOG DATASETS")
    print("=" * 70)

    np.save(BASE_DIR / "hog_X_all.npy", X_all)
    np.save(BASE_DIR / "hog_y_all.npy", y_all)
    np.save(BASE_DIR / "hog_X_test.npy", X_test)
    np.save(BASE_DIR / "hog_y_test.npy", y_test)
    np.save(
        BASE_DIR / "hog_train_paths.npy", np.array(valid_training_paths, dtype=object)
    )
    np.save(BASE_DIR / "hog_test_paths.npy", np.array(valid_test_paths, dtype=object))

    print("\nFiles successfully saved:")
    print(f"hog_X_all.npy        : {X_all.shape}")
    print(f"hog_y_all.npy        : {y_all.shape}")
    print(f"hog_X_test.npy       : {X_test.shape}")
    print(f"hog_y_test.npy       : {y_test.shape}")
    print(f"hog_train_paths.npy  : {len(valid_training_paths)} paths")
    print(f"hog_test_paths.npy   : {len(valid_test_paths)} paths")

    print("\n" + "=" * 70)
    print("TASK 1 HOG FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Training images processed : {len(X_all)}")
    print(f"Testing images processed  : {len(X_test)}")
    print(f"HOG features per image    : {X_all.shape[1]}")
    print(f"Training matrix           : {X_all.shape}")
    print(f"Testing matrix            : {X_test.shape}")
    print(f"Training failures         : {len(failed_training)}")
    print(f"Testing failures          : {len(failed_test)}")

    training_paths = [Path(p) for p in valid_training_paths]
    test_paths_out = [Path(p) for p in valid_test_paths]

    return X_all, y_all, X_test, y_test, training_paths, test_paths_out, display_results


def show_hog_visualisation(display_results):
    """Pop up the four-panel HOG visualisation for a handful of test images."""
    print("\n" + "=" * 70)
    print("HOG FEATURE VISUALISATION")
    print("=" * 70)
    print(f"Displaying {len(display_results)} representative test images.")
    print("Press any key for next image.")
    print("Press Esc to stop.")

    cv2.namedWindow(HOG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    for result in display_results:
        display = create_hog_display(*result)
        cv2.imshow(HOG_WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


# ============================================================
# PART 2: RANDOM FOREST CLASSIFICATION (was random_forest.py)
# ============================================================


def calculate_metrics(y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(y_true, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)
    return accuracy, precision, recall, f1


def create_random_forest():
    return RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)


def train_and_validate(X_all, y_all):
    print("\n" + "=" * 65)
    print("TRAIN / VALIDATION EXPERIMENT")
    print("=" * 65)

    indices = np.arange(len(X_all))
    train_indices, validation_indices = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_SEED, stratify=y_all
    )

    X_train, X_validation = X_all[train_indices], X_all[validation_indices]
    y_train, y_validation = y_all[train_indices], y_all[validation_indices]

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_validation)}")

    model = create_random_forest()

    print("\nTraining Random Forest...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    validation_predictions = model.predict(X_validation)
    accuracy, precision, recall, f1 = calculate_metrics(
        y_validation, validation_predictions
    )

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score  : {f1:.4f} ({f1 * 100:.2f}%)")

    print("\nGenerating validation confusion matrix...")
    cm = confusion_matrix(y_validation, validation_predictions)
    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, xticks_rotation="vertical")
    ax.set_title("HOG + Random Forest - Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "random_forest_validation_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()
    print("Saved: random_forest_validation_confusion_matrix.png")

    return model, train_indices, validation_indices


def create_prediction_display(
    original, cropped_sign, filename, actual, predicted, dataset_type
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

    panels = cv2.hconcat([original_panel, cropped_panel, result_panel])
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


def visualise_results(image_paths, labels, predictions, indices, dataset_type):
    print("\n" + "=" * 65)
    print(f"VISUALISING {len(indices)} {dataset_type.upper()} RESULTS")
    print("=" * 65)
    print("Press any key for next image.")
    print("Press Esc to stop.")

    window_name = f"Task 2 - Random Forest - {dataset_type}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for number, index in enumerate(indices, start=1):
        image_path = image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        cropped_sign = extract_sign_region(image)
        actual, predicted = int(labels[index]), int(predictions[index])

        display = create_prediction_display(
            original=image,
            cropped_sign=cropped_sign,
            filename=image_path.name,
            actual=actual,
            predicted=predicted,
            dataset_type=f"{dataset_type} {number}/{len(indices)}",
        )
        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            print("Visualisation stopped by user.")
            break

    cv2.destroyWindow(window_name)


def visualise_training_results(model, X_all, y_all, training_paths, train_indices):
    print("\nPreparing training result visualisation...")

    all_predictions = model.predict(X_all)
    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.choice(
        train_indices, size=min(NUM_VISUAL_TRAINING, len(train_indices)), replace=False
    )

    print(f"Selected {len(selected_indices)} training images.")

    visualise_results(
        image_paths=training_paths,
        labels=y_all,
        predictions=all_predictions,
        indices=selected_indices,
        dataset_type="Training",
    )


def train_final_model(X_all, y_all):
    print("\n" + "=" * 65)
    print("FINAL RANDOM FOREST TRAINING")
    print("=" * 65)

    model = create_random_forest()
    print(f"Training final model using all {len(X_all)} training images...")

    start_time = time.time()
    model.fit(X_all, y_all)
    training_time = time.time() - start_time
    print(f"Final training completed in {training_time:.2f} seconds.")

    return model


def test_random_forest(model, X_test, y_test):
    print("\n" + "=" * 65)
    print("FINAL TESTING - LECTURER TEST IMAGES")
    print("=" * 65)

    print("\nMaking predictions...")
    start_time = time.time()
    predictions = model.predict(X_test)
    prediction_time = time.time() - start_time

    accuracy, precision, recall, f1 = calculate_metrics(y_test, predictions)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score  : {f1:.4f} ({f1 * 100:.2f}%)")

    correct_predictions = np.sum(predictions == y_test)
    print(f"\nCorrect predictions: {correct_predictions}/{len(y_test)}")
    print(f"Prediction time: {prediction_time:.4f} seconds")

    print("\nGenerating final test confusion matrix...")
    cm_test = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(
        ax=ax, xticks_rotation="vertical"
    )
    ax.set_title("HOG + Random Forest - Final Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "random_forest_test_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()
    print("Saved: random_forest_test_confusion_matrix.png")

    return predictions


def visualise_selected_test_results(test_paths, y_test, predictions):
    selected_indices = np.linspace(
        0, len(y_test) - 1, num=min(NUM_VISUAL_TEST, len(y_test)), dtype=int
    )

    visualise_results(
        image_paths=test_paths,
        labels=y_test,
        predictions=predictions,
        indices=selected_indices,
        dataset_type="Test",
    )

    return selected_indices


def visualise_remaining_test_results(test_paths, y_test, predictions, selected_indices):
    """
    Show the leftover test images (everything not in the 10 spot-checked
    above) as grid pages of 10 images each, for easy screenshotting into
    a report appendix. No cropped-sign panel, no per-image comparison
    popup -- just the image with an actual/predicted/status caption.
    Each page is also auto-saved as a .jpg.
    """
    selected_set = set(selected_indices.tolist())
    remaining_indices = [
        index for index in range(len(y_test)) if index not in selected_set
    ]

    print(f"\nRemaining test images for Appendix: {len(remaining_indices)}")

    images_per_page = 10
    rows, cols = 2, 5
    num_pages = math.ceil(len(remaining_indices) / images_per_page)

    output_dir = BASE_DIR / "appendix_test_results"
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {num_pages} appendix page(s), {images_per_page} images each...")

    for page in range(num_pages):
        start = page * images_per_page
        end = min(start + images_per_page, len(remaining_indices))
        page_indices = remaining_indices[start:end]

        fig, axes = plt.subplots(rows, cols, figsize=(15, 7), facecolor="white")
        axes = axes.flatten()

        for position, index in enumerate(page_indices):
            image_path = test_paths[index]
            image = cv2.imread(str(image_path))

            if image is None:
                axes[position].axis("off")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            actual = int(y_test[index])
            predicted = int(predictions[index])
            status = "CORRECT" if actual == predicted else "WRONG"

            axes[position].imshow(image)
            axes[position].axis("off")
            axes[position].set_title(
                f"Actual: {actual:03d}\nPredicted: {predicted:03d}\n{status}",
                fontsize=9,
                fontweight="bold",
            )

        for position in range(len(page_indices), len(axes)):
            axes[position].axis("off")

        plt.tight_layout()

        output_path = output_dir / f"appendix_test_results_page_{page + 1:02d}.jpg"
        fig.savefig(output_path, dpi=200, facecolor="white", bbox_inches="tight")
        print(f"Saved: {output_path.name}")

        plt.show()
        plt.close(fig)

    print(f"\nAll appendix pages saved to: {output_dir.resolve()}")


def print_final_summary(final_model, X_all, y_all, X_test, y_test, test_predictions):
    train_predictions = final_model.predict(X_all)
    train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(
        y_all, train_predictions
    )
    test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(
        y_test, test_predictions
    )

    print("\n" + "=" * 75)
    print("FINAL PERFORMANCE COMPARISON - HOG + RANDOM FOREST")
    print("=" * 75)
    print(
        f"{'Dataset':<15}{'Accuracy':<15}{'Precision':<15}{'Recall':<15}{'F1-score':<15}"
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
    # ---- Task 1: HOG feature extraction ----
    (
        X_all,
        y_all,
        X_test,
        y_test,
        training_paths,
        test_paths,
        hog_display_results,
    ) = extract_hog_features()

    if hog_display_results:
        show_hog_visualisation(hog_display_results)

    if len(training_paths) != len(X_all):
        raise ValueError(
            f"Training paths ({len(training_paths)}) do not match HOG features ({len(X_all)})."
        )
    if len(test_paths) != len(X_test):
        raise ValueError(
            f"Test paths ({len(test_paths)}) do not match HOG features ({len(X_test)})."
        )

    # ---- Task 2: Random Forest classification ----
    print("\n" + "=" * 70)
    print("TASK 2: RANDOM FOREST CLASSIFICATION USING HOG FEATURES")
    print("=" * 70)

    validation_model, train_indices, validation_indices = train_and_validate(
        X_all, y_all
    )

    print("\n" + "#" * 65)
    print("STEP 1: VISUALISE 10 TRAINING RESULTS")
    print("#" * 65)
    visualise_training_results(
        validation_model, X_all, y_all, training_paths, train_indices
    )

    final_model = train_final_model(X_all, y_all)
    test_predictions = test_random_forest(final_model, X_test, y_test)

    print("\n" + "#" * 65)
    print("STEP 2: VISUALISE 10 TEST RESULTS")
    print("#" * 65)
    selected_test_indices = visualise_selected_test_results(
        test_paths, y_test, test_predictions
    )

    print("\n" + "#" * 65)
    print("STEP 3: REMAINING TEST RESULTS FOR APPENDIX")
    print("#" * 65)
    visualise_remaining_test_results(
        test_paths, y_test, test_predictions, selected_test_indices
    )

    print_final_summary(final_model, X_all, y_all, X_test, y_test, test_predictions)


if __name__ == "__main__":
    main()
