"""Task 1: HOG Feature Extraction only."""

import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TRAIN_DIR = Path(
    r"C:\Users\teeak\Desktop\Study\Y3S1"
    r"\Mini Project (Assignment 2)\Training"
)

ANNOTATION_FILE = TRAIN_DIR / "annotations.csv"

IMAGE_SIZE = (64, 64)
DISPLAY_COUNT = 10


# ---------------------------------------------------------
# Read and clean annotations
# ---------------------------------------------------------

def read_annotations():
    annotations_by_filename = {}

    with ANNOTATION_FILE.open(
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            filename = row["file_name"].strip()
            category = int(float(row["category"]))
            image_path = TRAIN_DIR / filename

            if not image_path.exists():
                continue

            annotation = {
                "category": category,
                "x1": int(float(row["x1"])),
                "y1": int(float(row["y1"])),
                "x2": int(float(row["x2"])),
                "y2": int(float(row["y2"])),
            }

            # Keep only one annotation for each filename
            annotations_by_filename.setdefault(filename, annotation)

    return annotations_by_filename


def crop_annotated_sign(image, annotation):
    """Crop the target traffic sign using its annotation bounding box."""
    image_height, image_width = image.shape[:2]

    x1 = max(0, min(annotation["x1"], image_width - 1))
    y1 = max(0, min(annotation["y1"], image_height - 1))
    x2 = max(x1 + 1, min(annotation["x2"], image_width))
    y2 = max(y1 + 1, min(annotation["y2"], image_height))

    cropped_sign = image[y1:y2, x1:x2]

    if cropped_sign.size == 0:
        return image

    return cropped_sign


# ---------------------------------------------------------
# Task 1: HOG feature extraction
# ---------------------------------------------------------

def extract_hog(image, create_visualisation=False):
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Standardise image size
    gray = cv2.resize(
        gray,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    result = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=create_visualisation,
        feature_vector=True
    )

    if create_visualisation:
        features, hog_image = result

        hog_image = cv2.normalize(
            hog_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return features.astype(np.float32), gray, hog_image

    return result.astype(np.float32)


# ---------------------------------------------------------
# Prepare popup visualisation
# ---------------------------------------------------------

def add_title(image, title):
    panel = cv2.resize(image, (300, 300))

    title_area = np.zeros(
        (45, panel.shape[1], 3),
        dtype=np.uint8
    )

    cv2.putText(
        title_area,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return cv2.vconcat([title_area, panel])


def create_display(image, cropped_sign, gray, hog_image, filename, category):
    original_panel = add_title(
        image,
        "Original"
    )

    cropped_panel = add_title(
        cropped_sign,
        "Cropped Sign"
    )

    grayscale_panel = add_title(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "Grayscale"
    )

    hog_panel = add_title(
        cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR),
        "HOG Features"
    )

    panels = cv2.hconcat([
        original_panel,
        cropped_panel,
        grayscale_panel,
        hog_panel
    ])

    header = np.zeros(
        (70, panels.shape[1], 3),
        dtype=np.uint8
    )

    cv2.putText(
        header,
        f"File: {filename} | Class: {category:03d}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
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


# ---------------------------------------------------------
# Run Task 1
# ---------------------------------------------------------

def main():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"annotations.csv was not found: {ANNOTATION_FILE}"
        )

    annotations_by_filename = read_annotations()

    feature_list = []
    label_list = []
    examples = []
    example_classes = set()

    print("=" * 65)
    print("TASK 1: HOG FEATURE EXTRACTION")
    print("=" * 65)
    print(f"Unique annotated images: {len(annotations_by_filename)}")
    print("Extracting HOG features...")

    for filename, annotation in annotations_by_filename.items():
        category = annotation["category"]
        image_path = TRAIN_DIR / filename
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {filename}")
            continue

        # Crop the labelled traffic sign before extracting HOG features
        cropped_sign = crop_annotated_sign(image, annotation)
        features = extract_hog(cropped_sign)

        feature_list.append(features)
        label_list.append(category)

        # Select 10 examples from different classes
        if (
            len(examples) < DISPLAY_COUNT
            and category not in example_classes
        ):
            _, gray, hog_image = extract_hog(
                cropped_sign,
                create_visualisation=True
            )

            examples.append(
                (
                    image,
                    cropped_sign,
                    gray,
                    hog_image,
                    filename,
                    category
                )
            )

            example_classes.add(category)

    # Complete HOG feature matrix
    X = np.vstack(feature_list)
    y = np.asarray(label_list)

    print("\nHOG extraction completed.")
    print(f"Images successfully processed : {len(X)}")
    print(f"HOG features per image        : {X.shape[1]}")
    print(f"HOG feature matrix shape      : {X.shape}")
    print(f"Label array shape             : {y.shape}")
    print(f"Visualisation examples        : {len(examples)}")
    print("\nPress any key for the next image.")
    print("Press Esc to stop.")

    cv2.namedWindow(
        "Task 1 - HOG Feature Extraction",
        cv2.WINDOW_NORMAL
    )

    for image, cropped_sign, gray, hog_image, filename, category in examples:
        display = create_display(
            image,
            cropped_sign,
            gray,
            hog_image,
            filename,
            category
        )

        cv2.imshow(
            "Task 1 - HOG Feature Extraction",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
