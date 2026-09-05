"""Task 1: HOG extraction on lecturer-provided testing images.

All testing images are processed and ten representative results are displayed.
No classifier, training, or class prediction is used in this Task 1 program.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TEST_DIR = Path(
    r"C:\Users\teeak\Desktop\Study\Y3S1"
    r"\Mini Project (Assignment 2)\Testing Image"
)

IMAGE_SIZE = (64, 64)
DISPLAY_COUNT = 20
EXPECTED_HOG_LENGTH = 1764
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
WINDOW_NAME = "Task 1 - HOG Feature Extraction"


# ---------------------------------------------------------
# Automatic traffic-sign region extraction
# ---------------------------------------------------------

def make_colour_mask(image):
    """Detect common red, blue and yellow traffic-sign colours."""
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
    """Automatically crop the most likely sign, or return the full image."""
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

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)

    cropped_sign = image[y1:y2, x1:x2]
    return cropped_sign if cropped_sign.size else image


# ---------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------

def extract_hog(image, create_visualisation=False):
    """Extract a 1,764-value HOG vector from one traffic-sign image."""
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
        feature_vector=True
    )

    if not create_visualisation:
        return result.astype(np.float32)

    features, hog_image = result
    hog_image = cv2.normalize(
        hog_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return features.astype(np.float32), gray, hog_image


def valid_hog_vector(features):
    return (
        features.ndim == 1
        and len(features) == EXPECTED_HOG_LENGTH
        and np.all(np.isfinite(features))
    )


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
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return cv2.vconcat([title_area, panel])


def create_display(image, cropped_sign, gray, hog_image, filename, category):
    panels = cv2.hconcat([
        add_title(image, "Original"),
        add_title(cropped_sign, "Extracted Sign"),
        add_title(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale"),
        add_title(cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR), "HOG Features")
    ])

    header = np.zeros((70, panels.shape[1], 3), dtype=np.uint8)

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
    test_paths = sorted(
        path for path in TEST_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_paths:
        raise FileNotFoundError(f"No testing images were found: {TEST_DIR}")

    selected_indices = set(
        np.linspace(
            0,
            len(test_paths) - 1,
            num=min(DISPLAY_COUNT, len(test_paths)),
            dtype=int
        ).tolist()
    )

    feature_list = []
    label_list = []
    failed_files = []
    display_results = []

    print("=" * 68)
    print("TASK 1: HOG EXTRACTION ON LECTURER TESTING IMAGES")
    print("=" * 68)
    print(f"Lecturer testing images          : {len(test_paths)}")
    print(f"Images selected for visualisation: {len(selected_indices)}")
    print(f"Standard image size              : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print(f"Expected HOG features per image  : {EXPECTED_HOG_LENGTH}")
    print("Region extraction                : Automatic colour/contour method")
    print("Classifier                       : Not used in Task 1")

    for index, image_path in enumerate(test_paths):
        image = cv2.imread(str(image_path))

        if image is None:
            failed_files.append((image_path.name, "unreadable image"))
            continue

        try:
            cropped_sign = extract_sign_region(image)

            if index in selected_indices:
                features, gray, hog_image = extract_hog(
                    cropped_sign,
                    create_visualisation=True
                )
                display_results.append(
                    (
                        image,
                        cropped_sign,
                        gray,
                        hog_image,
                        image_path.name,
                        int(image_path.name.split("_")[0])
                    )
                )
            else:
                features = extract_hog(cropped_sign)

            if not valid_hog_vector(features):
                raise ValueError("invalid HOG feature vector")

            feature_list.append(features)
            label_list.append(int(image_path.name.split("_")[0]))

        except (cv2.error, ValueError) as error:
            failed_files.append((image_path.name, str(error)))

    if not feature_list:
        raise RuntimeError("No valid HOG features were extracted.")

    X = np.vstack(feature_list)
    y = np.asarray(label_list)
    technical_rate = len(X) / len(test_paths)

    print("\n" + "=" * 68)
    print("HOG FEATURE EXTRACTION RESULTS")
    print("=" * 68)
    print(f"Successfully extracted images    : {len(X)}/{len(test_paths)}")
    print(f"Technical processing rate        : {technical_rate * 100:.2f}%")
    print(f"HOG features per image           : {X.shape[1]}")
    print(f"HOG feature matrix shape         : {X.shape}")
    print(f"Label array shape                : {y.shape}")
    print(f"Visualisation results            : {len(display_results)}")
    print("Classifier                       : Not used")
    print(
        "Visual review rate              : Count the acceptable results "
        f"among the {len(display_results)} displayed images"
    )

    if failed_files:
        print("\nFailed files:")
        for filename, reason in failed_files:
            print(f"- {filename}: {reason}")

    print("\nPress any key to view the next visualisation.")
    print("Press Esc to stop.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for result in display_results:
        cv2.imshow(WINDOW_NAME, create_display(*result))

        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
