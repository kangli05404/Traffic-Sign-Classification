from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TEST_DIR = Path(
    r"C:\Users\wenji\Downloads\MP - A2"
    r"\test_images"
)

IMAGE_SIZE = (64, 64)
DISPLAY_COUNT = 20
EXPECTED_LBP_LENGTH = 640
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
WINDOW_NAME = "Task 1 - Spatial LBP Feature Extraction"


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
# Spatial LBP feature extraction
# ---------------------------------------------------------

def extract_lbp(image, create_visualisation=False):
    """Extract a 640-value Spatial LBP vector from one testing image."""
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    lbp_image = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    cells_per_side = IMAGE_SIZE[0] // 8

    cells = lbp_image.reshape(
        cells_per_side,
        8,
        cells_per_side,
        8
    ).transpose(0, 2, 1, 3)

    histograms = np.stack(
        [
            (cells == pattern).sum(axis=(2, 3))
            for pattern in range(8 + 2)
        ],
        axis=-1
    ).astype(np.float32)

    histograms /= histograms.sum(axis=-1, keepdims=True) + 1e-7
    features = histograms.reshape(-1)

    if not create_visualisation:
        return features.astype(np.float32)

    lbp_visual = cv2.normalize(
        lbp_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return features.astype(np.float32), gray, lbp_visual


def valid_lbp_vector(features):
    """Confirm that a complete Spatial LBP vector was produced."""
    return (
        features.ndim == 1
        and len(features) == EXPECTED_LBP_LENGTH
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


def create_display(
    image,
    cropped_sign,
    gray,
    lbp_image,
    filename,
    review_number,
    total_reviews
):
    original_panel = add_title(image, "Original")
    cropped_panel = add_title(cropped_sign, "Extracted Sign")
    grayscale_panel = add_title(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "Grayscale"
    )
    lbp_panel = add_title(
        cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2BGR),
        "Spatial LBP"
    )

    panels = cv2.hconcat([
        original_panel,
        cropped_panel,
        grayscale_panel,
        lbp_panel
    ])

    header = np.zeros((85, panels.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        header,
        f"File: {filename} | Review: {review_number}/{total_reviews}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        header,
        "C = Correct extraction | X = Incorrect extraction | Esc = stop",
        (15, 62),
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
        path
        for path in TEST_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_paths:
        raise RuntimeError(
            f"No valid testing images were found in: {TEST_DIR}"
        )

    selected_indices = set(
        np.linspace(
            0,
            len(test_paths) - 1,
            num=min(DISPLAY_COUNT, len(test_paths)),
            dtype=int
        ).tolist()
    )

    feature_list = []
    failed_files = []
    display_results = []

    print("=" * 68)
    print("TASK 1: SPATIAL LBP FEATURE EXTRACTION AND VISUALISATION")
    print("=" * 68)
    print(f"Testing images                   : {len(test_paths)}")
    print(f"Images selected for visualisation: {len(selected_indices)}")
    print(f"Standard image size              : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print("LBP configuration                : P=8, R=1, uniform")
    print("Spatial grid                     : 8 x 8 cells")
    print("Cell size                        : 8 x 8 pixels")
    print(f"Expected LBP features per image  : {EXPECTED_LBP_LENGTH}")
    print("Region extraction                : Automatic colour/contour method")
    print("Classifier                       : Not used in Task 1")
    print("\nExtracting Spatial LBP features from the testing dataset...")

    for index, image_path in enumerate(test_paths):
        image = cv2.imread(str(image_path))

        if image is None:
            failed_files.append(
                (image_path.name, "unreadable image")
            )
            continue

        try:
            cropped_sign = extract_sign_region(image)

            if index in selected_indices:
                features, gray, lbp_image = extract_lbp(
                    cropped_sign,
                    create_visualisation=True
                )

                display_results.append(
                    (
                        image,
                        cropped_sign,
                        gray,
                        lbp_image,
                        image_path.name
                    )
                )
            else:
                features = extract_lbp(cropped_sign)

            if not valid_lbp_vector(features):
                raise ValueError(
                    "invalid Spatial LBP feature vector"
                )

            feature_list.append(features)

        except (cv2.error, ValueError) as error:
            failed_files.append(
                (image_path.name, str(error))
            )

    if not feature_list:
        raise RuntimeError(
            "No valid Spatial LBP features were extracted."
        )

    X = np.vstack(feature_list)
    technical_rate = len(X) / len(test_paths)

    print("\n" + "=" * 68)
    print("SPATIAL LBP FEATURE EXTRACTION RESULTS")
    print("=" * 68)
    print(f"Successfully extracted images    : {len(X)}/{len(test_paths)}")
    print(f"Technical processing rate        : {technical_rate * 100:.2f}%")
    print(f"LBP features per image           : {X.shape[1]}")
    print(f"LBP feature matrix shape         : {X.shape}")
    print(f"Visualisation results            : {len(display_results)}")
    print("Classifier                       : Not used")

    if failed_files:
        print("\nFailed files:")

        for filename, reason in failed_files:
            print(f"- {filename}: {reason}")

    print("\nVisual review")
    print("C = Correct feature extraction")
    print("X = Incorrect feature extraction")
    print("Esc = Stop review")

    correct_count = 0
    incorrect_count = 0
    reviewed_results = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    total_reviews = len(display_results)

    for review_number, result in enumerate(
        display_results,
        start=1
    ):
        image, cropped_sign, gray, lbp_image, filename = result

        display = create_display(
            image,
            cropped_sign,
            gray,
            lbp_image,
            filename,
            review_number,
            total_reviews
        )

        cv2.imshow(WINDOW_NAME, display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("c"), ord("C")):
                correct_count += 1
                reviewed_results.append(
                    (filename, "Correct")
                )
                break

            if key in (ord("x"), ord("X")):
                incorrect_count += 1
                reviewed_results.append(
                    (filename, "Incorrect")
                )
                break

            if key == 27:
                cv2.destroyAllWindows()
                break

        if key == 27:
            break

    cv2.destroyAllWindows()

    reviewed_total = correct_count + incorrect_count

    print("\n" + "=" * 68)
    print("VISUAL REVIEW RESULTS")
    print("=" * 68)

    for filename, result in reviewed_results:
        print(f"{filename:<28} {result}")

    print(f"\nImages reviewed                  : {reviewed_total}")
    print(f"Correct feature extraction       : {correct_count}")
    print(f"Incorrect feature extraction     : {incorrect_count}")

    if reviewed_total > 0:
        review_rate = correct_count / reviewed_total
        print(
            f"Visual review success rate       : "
            f"{review_rate * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
