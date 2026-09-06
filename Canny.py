"""Task 1: Canny extraction on training, validation and testing images."""

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "Training"
ANNOTATION_FILE = TRAIN_DIR / "annotations.csv"
TEST_DIR = PROJECT_DIR / "Testing Image"

IMAGE_SIZE = (64, 64)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
EXPECTED_CANNY_LENGTH = 4096
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42
DISPLAY_ALLOCATION = {"Training": 4, "Validation": 3, "Testing": 3}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
WINDOW_NAME = "Task 1 - Canny Edge Features"

# ---------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------

def read_annotations():
    labels_by_filename = {}
    raw_row_count = 0

    with ANNOTATION_FILE.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            raw_row_count += 1
            filename = row["file_name"].strip()
            image_path = TRAIN_DIR / filename

            if image_path.exists():
                labels_by_filename.setdefault(
                    filename,
                    int(float(row["category"]))
                )

    return labels_by_filename, raw_row_count


def image_hash(path):
    """Return a SHA-256 hash used to find exact image duplicates."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def prepare_partitions():
    """Create clean training, validation and lecturer testing partitions."""
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"annotations.csv was not found: {ANNOTATION_FILE}"
        )

    test_paths = sorted(
        path for path in TEST_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not test_paths:
        raise FileNotFoundError(f"No testing images were found: {TEST_DIR}")

    labels_by_filename, raw_row_count = read_annotations()
    test_hashes = {image_hash(path) for path in test_paths}
    clean_records = sorted(
        (
            (TRAIN_DIR / filename, category)
            for filename, category in labels_by_filename.items()
            if image_hash(TRAIN_DIR / filename) not in test_hashes
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
        "clean_images": len(clean_records),
    }
    return {
        "Training": training_records,
        "Validation": validation_records,
        "Testing": testing_records,
    }, metadata


def representative_indices(total, count):
    """Return evenly distributed deterministic indices."""
    if total == 0 or count == 0:
        return set()

    return set(np.linspace(
        0,
        total - 1,
        num=min(count, total),
        dtype=int
    ).tolist())


# ---------------------------------------------------------
# Automatic traffic-sign region extraction
# ---------------------------------------------------------

def make_colour_mask(image):
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
# Canny edge feature extraction
# ---------------------------------------------------------

def extract_canny_features(image):
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edge_image = cv2.Canny(
        blurred,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD
    )
    features = edge_image.astype(np.float32).reshape(-1) / 255.0
    return features, gray, edge_image


def valid_canny_vector(features):
    return (
        features.ndim == 1
        and len(features) == EXPECTED_CANNY_LENGTH
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
    edge_image,
    filename,
    category,
    partition
):
    panels = cv2.hconcat([
        add_title(image, "Original"),
        add_title(cropped_sign, "Extracted Sign"),
        add_title(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale"),
        add_title(
            cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR),
            "Canny Edge Features"
        )
    ])
    header = np.zeros((70, panels.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"Split: {partition} | File: {filename} | Class: {category:03d}",
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
# Partition processing
# ---------------------------------------------------------

def process_partition(partition, records, display_count):
    """Extract Canny features from every record in one partition."""
    selected_indices = representative_indices(len(records), display_count)
    feature_list = []
    label_list = []
    failed_files = []
    display_results = []
    empty_edge_count = 0

    for index, (image_path, category) in enumerate(records):
        image = cv2.imread(str(image_path))
        if image is None:
            failed_files.append((image_path.name, "unreadable image"))
            continue

        try:
            cropped_sign = extract_sign_region(image)
            features, gray, edge_image = extract_canny_features(cropped_sign)

            if not valid_canny_vector(features):
                raise ValueError("invalid Canny edge vector")

            if np.count_nonzero(features) == 0:
                empty_edge_count += 1

            feature_list.append(features)
            label_list.append(category)

            if index in selected_indices:
                display_results.append((
                    image,
                    cropped_sign,
                    gray,
                    edge_image,
                    image_path.name,
                    category,
                    partition
                ))
        except (cv2.error, ValueError) as error:
            failed_files.append((image_path.name, str(error)))

    if not feature_list:
        raise RuntimeError(f"No valid Canny features for {partition}.")

    return {
        "X": np.vstack(feature_list),
        "y": np.asarray(label_list),
        "total": len(records),
        "failed": failed_files,
        "displays": display_results,
        "empty_edges": empty_edge_count,
    }


def main():
    partitions, metadata = prepare_partitions()

    print("=" * 72)
    print("TASK 1: CANNY EXTRACTION ON TRAINING, VALIDATION AND TESTING DATA")
    print("=" * 72)
    print(f"Standard image size              : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print(
        f"Canny thresholds                 : {CANNY_LOW_THRESHOLD}, "
        f"{CANNY_HIGH_THRESHOLD}"
    )
    print(f"Expected Canny features per image: {EXPECTED_CANNY_LENGTH}")
    print("Region extraction                : Automatic colour/contour method")
    print("Extracting Canny features...")

    results = {
        name: process_partition(name, records, DISPLAY_ALLOCATION[name])
        for name, records in partitions.items()
    }

    print("\n" + "=" * 72)
    print("CANNY EDGE FEATURE EXTRACTION RESULTS")
    print("=" * 72)
    all_displays = []
    all_failures = []

    for name, result in results.items():
        successful = len(result["X"])
        print(f"\n{name}")
        print(f"  Successfully extracted         : {successful}/{result['total']}")
        print(f"  Feature matrix shape           : {result['X'].shape}")
        print(f"  Visualisation samples          : {len(result['displays'])}")
        all_displays.extend(result["displays"])
        all_failures.extend(
            (name, filename, reason)
            for filename, reason in result["failed"]
        )

    total_successful = sum(len(result["X"]) for result in results.values())
    total_images = sum(result["total"] for result in results.values())
    print("\nOverall")
    print(f"  Successfully extracted         : {total_successful}/{total_images}")
    print(f"  Total visualisation samples    : {len(all_displays)}")
    print("  Visual review rate             : Count acceptable results among "
          f"the {len(all_displays)} displayed images")

    if all_failures:
        print("\nFailed files:")
        for name, filename, reason in all_failures:
            print(f"- [{name}] {filename}: {reason}")

    print("\nPress any key to view the next visualisation.")
    print("Press Esc to stop.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for result in all_displays:
        cv2.imshow(WINDOW_NAME, create_display(*result))
        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
