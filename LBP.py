import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TRAIN_DIR = Path(
    r"C:\Users\wenji\Downloads\MP - A2"
    r"\training"
)
TRAIN_IMAGE_DIR = TRAIN_DIR / "images"
ANNOTATION_FILE = TRAIN_DIR / "annotations.csv"

TEST_DIR = Path(
    r"C:\Users\wenji\Downloads\MP - A2"
    r"\test_images"
)

IMAGE_SIZE = (64, 64)
EXPECTED_LBP_LENGTH = 640
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42
DISPLAY_ALLOCATION = {"Training": 4, "Validation": 3, "Testing": 3}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
WINDOW_NAME = "Task 1 - Spatial LBP Feature Extraction"


# ---------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------

def read_annotations():
    """Read labels and keep one annotation for each image filename."""
    labels_by_filename = {}
    raw_row_count = 0

    with ANNOTATION_FILE.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            raw_row_count += 1
            filename = row["file_name"].strip()
            image_path = TRAIN_IMAGE_DIR / filename

            if image_path.exists():
                labels_by_filename.setdefault(
                    filename,
                    int(float(row["category"]))
                )

    return labels_by_filename, raw_row_count


def image_hash(path):
    """Return SHA-256 hash for exact duplicate checking."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def prepare_partitions():
    """Prepare training, validation and testing partitions."""
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"annotations.csv was not found: {ANNOTATION_FILE}"
        )

    test_paths = sorted(
        path for path in TEST_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_paths:
        raise FileNotFoundError(
            f"No testing images were found: {TEST_DIR}"
        )

    labels_by_filename, raw_row_count = read_annotations()
    test_hashes = {image_hash(path) for path in test_paths}

    clean_records = sorted(
        (
            (TRAIN_IMAGE_DIR / filename, category)
            for filename, category in labels_by_filename.items()
            if image_hash(TRAIN_IMAGE_DIR / filename) not in test_hashes
        ),
        key=lambda item: item[0].name.lower()
    )

    training_records, validation_records = train_test_split(
        clean_records,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=[category for _, category in clean_records]
    )

    training_records.sort(key=lambda item: item[0].name.lower())
    validation_records.sort(key=lambda item: item[0].name.lower())

    testing_records = [
        (path, int(path.name.split("_")[0]))
        for path in test_paths
    ]

    metadata = {
        "raw_rows": raw_row_count,
        "unique_images": len(labels_by_filename),
        "overlaps_removed": len(labels_by_filename) - len(clean_records),
        "clean_images": len(clean_records)
    }

    partitions = {
        "Training": training_records,
        "Validation": validation_records,
        "Testing": testing_records
    }

    return partitions, metadata


def representative_indices(total, count):
    """Select evenly distributed images for visualisation."""
    if total == 0 or count == 0:
        return set()

    return set(
        np.linspace(
            0,
            total - 1,
            num=min(count, total),
            dtype=int
        ).tolist()
    )


# ---------------------------------------------------------
# Automatic traffic-sign region extraction
# ---------------------------------------------------------

def make_colour_mask(image):
    """Detect red, blue and yellow traffic-sign colours."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 40), (10, 255, 255)),
        cv2.inRange(hsv, (170, 70, 40), (180, 255, 255))
    )
    blue = cv2.inRange(hsv, (90, 60, 35), (140, 255, 255))
    yellow = cv2.inRange(hsv, (15, 70, 45), (40, 255, 255))

    mask = cv2.bitwise_or(cv2.bitwise_or(red, blue), yellow)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )


def extract_sign_region(image):
    """Automatically crop the most likely traffic sign."""
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
    """Extract a 640-value Spatial LBP feature vector."""
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
            for pattern in range(10)
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
    """Check whether the LBP feature vector is valid."""
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
    category,
    partition,
    review_number,
    total_reviews
):
    panels = cv2.hconcat([
        add_title(image, "Original"),
        add_title(cropped_sign, "Extracted Sign"),
        add_title(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            "Grayscale"
        ),
        add_title(
            cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2BGR),
            "Spatial LBP"
        )
    ])

    header = np.zeros(
        (85, panels.shape[1], 3),
        dtype=np.uint8
    )

    cv2.putText(
        header,
        (
            f"Split: {partition} | "
            f"File: {filename} | "
            f"Class: {category:03d} | "
            f"Review: {review_number}/{total_reviews}"
        ),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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
# Partition processing
# ---------------------------------------------------------

def process_partition(partition, records, display_count):
    """Extract LBP features from all images in one partition."""
    selected_indices = representative_indices(
        len(records),
        display_count
    )

    feature_list = []
    label_list = []
    failed_files = []
    display_results = []

    for index, (image_path, category) in enumerate(records):
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

                display_results.append((
                    image,
                    cropped_sign,
                    gray,
                    lbp_image,
                    image_path.name,
                    category,
                    partition
                ))
            else:
                features = extract_lbp(cropped_sign)

            if not valid_lbp_vector(features):
                raise ValueError(
                    "invalid Spatial LBP feature vector"
                )

            feature_list.append(features)
            label_list.append(category)

        except (cv2.error, ValueError) as error:
            failed_files.append(
                (image_path.name, str(error))
            )

    if not feature_list:
        raise RuntimeError(
            f"No valid Spatial LBP features for {partition}."
        )

    return {
        "X": np.vstack(feature_list),
        "y": np.asarray(label_list),
        "total": len(records),
        "failed": failed_files,
        "displays": display_results
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    partitions, metadata = prepare_partitions()

    print("=" * 72)
    print(
        "TASK 1: SPATIAL LBP EXTRACTION ON "
        "TRAINING, VALIDATION AND TESTING DATA"
    )
    print("=" * 72)

    print(f"Raw annotation rows              : {metadata['raw_rows']}")
    print(f"Unique annotated images          : {metadata['unique_images']}")
    print(f"Exact testing overlaps removed   : {metadata['overlaps_removed']}")
    print(f"Clean images before split        : {metadata['clean_images']}")
    print(f"Training images                  : {len(partitions['Training'])}")
    print(f"Validation images                : {len(partitions['Validation'])}")
    print(f"Lecturer testing images          : {len(partitions['Testing'])}")
    print(f"Standard image size              : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print("LBP configuration                : P=8, R=1, uniform")
    print("Spatial grid                     : 8 x 8 cells")
    print("Cell size                        : 8 x 8 pixels")
    print(f"Expected LBP features per image  : {EXPECTED_LBP_LENGTH}")
    print("Region extraction                : Automatic colour/contour method")

    results = {
        name: process_partition(
            name,
            records,
            DISPLAY_ALLOCATION[name]
        )
        for name, records in partitions.items()
    }

    print("\n" + "=" * 72)
    print("SPATIAL LBP FEATURE EXTRACTION RESULTS")
    print("=" * 72)

    all_displays = []
    all_failures = []

    for name, result in results.items():
        successful = len(result["X"])
        rate = successful / result["total"]

        print(f"\n{name}")
        print(
            f"  Successfully extracted         : "
            f"{successful}/{result['total']}"
        )
        print(
            f"  Technical processing rate      : "
            f"{rate * 100:.2f}%"
        )
        print(f"  Feature matrix shape           : {result['X'].shape}")
        print(f"  Label array shape              : {result['y'].shape}")
        print(f"  Visualisation samples          : {len(result['displays'])}")

        all_displays.extend(result["displays"])

        all_failures.extend(
            (name, filename, reason)
            for filename, reason in result["failed"]
        )

    total_successful = sum(
        len(result["X"])
        for result in results.values()
    )

    total_images = sum(
        result["total"]
        for result in results.values()
    )

    print("\nOverall")
    print(
        f"  Successfully extracted         : "
        f"{total_successful}/{total_images}"
    )
    print(
        f"  Technical processing rate      : "
        f"{total_successful / total_images * 100:.2f}%"
    )
    print(
        f"  Spatial LBP features per image : "
        f"{EXPECTED_LBP_LENGTH}"
    )
    print(
        f"  Total visualisation samples    : "
        f"{len(all_displays)}"
    )

    if all_failures:
        print("\nFailed files:")

        for name, filename, reason in all_failures:
            print(f"- [{name}] {filename}: {reason}")

    # -----------------------------------------------------
    # Manual visual review
    # -----------------------------------------------------

    print("\nVisual review")
    print("C = Correct feature extraction")
    print("X = Incorrect feature extraction")
    print("Esc = Stop review")

    correct_count = 0
    incorrect_count = 0
    reviewed_results = []

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    total_reviews = len(all_displays)

    for review_number, result in enumerate(
        all_displays,
        start=1
    ):
        (
            image,
            cropped_sign,
            gray,
            lbp_image,
            filename,
            category,
            partition
        ) = result

        display = create_display(
            image,
            cropped_sign,
            gray,
            lbp_image,
            filename,
            category,
            partition,
            review_number,
            total_reviews
        )

        cv2.imshow(WINDOW_NAME, display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("c"), ord("C")):
                correct_count += 1

                reviewed_results.append(
                    (
                        partition,
                        filename,
                        category,
                        "Correct"
                    )
                )
                break

            if key in (ord("x"), ord("X")):
                incorrect_count += 1

                reviewed_results.append(
                    (
                        partition,
                        filename,
                        category,
                        "Incorrect"
                    )
                )
                break

            if key == 27:
                cv2.destroyAllWindows()
                break

        if key == 27:
            break

    cv2.destroyAllWindows()

    reviewed_total = correct_count + incorrect_count

    print("\n" + "=" * 72)
    print("VISUAL REVIEW RESULTS")
    print("=" * 72)

    for partition, filename, category, result in reviewed_results:
        print(
            f"{partition:<11} "
            f"{filename:<24} "
            f"Class={category:03d} "
            f"{result}"
        )

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