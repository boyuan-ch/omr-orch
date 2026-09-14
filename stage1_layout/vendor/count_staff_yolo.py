import numpy as np
import cv2

def count_staves_from_yolo(
    yolo_txt_path,
    img_path,
    target_label=1,
    black_ratio_threshold=0.02,
    black_pixel_thresh=40
):
    """
    Count staves and measures from a YOLO label file, considering only target_label.
    Remove boxes whose black pixel ratio is too low.

    Each YOLO line format:
        class x_center y_center width height confidence (normalized)

    Returns:
        n_staves: int
        staff_measure_counts: list[int]
        staff_height: float
        avg_start_x: float
        staff_y_centers: list[float]
        target_label_boxes_with_conf: list[list[float]]
    """

    # -----------------------------
    # Load image
    # -----------------------------
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    H, W = img.shape

    boxes = []  # [xc, yc, w, h]
    target_label_boxes_with_conf = []

    # -----------------------------
    # Read YOLO txt
    # -----------------------------
    with open(yolo_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue

            cls, xc, yc, bw, bh, conf = map(float, parts)

            if int(cls) != target_label:
                continue

            # -----------------------------
            # Convert normalized → pixel bbox
            # -----------------------------
            x1 = int((xc - bw / 2) * W)
            x2 = int((xc + bw / 2) * W)
            y1 = int((yc - bh / 2) * H)
            y2 = int((yc + bh / 2) * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = img[y1:y2, x1:x2]

            # -----------------------------
            # Compute black pixel ratio
            # -----------------------------
            black_pixels = np.sum(roi < black_pixel_thresh)
            black_ratio = black_pixels / roi.size

            if black_ratio < black_ratio_threshold:
                continue  # ❌ discard weak box

            # -----------------------------
            # Keep valid box
            # -----------------------------
            boxes.append([xc, yc, bw, bh])
            target_label_boxes_with_conf.append([xc, yc, bw, bh, conf])

    if not boxes:
        raise ValueError(f"No valid boxes after filtering in {yolo_txt_path}")

    # -----------------------------
    # Original logic (unchanged)
    # -----------------------------
    heights = [b[3] for b in boxes]
    staff_height = np.mean(heights)

    boxes.sort(key=lambda b: b[1])

    staff_groups = []
    current_group = [boxes[0]]

    for b in boxes[1:]:
        if abs(b[1] - current_group[-1][1]) < staff_height:
            current_group.append(b)
        else:
            staff_groups.append(current_group)
            current_group = [b]

    staff_groups.append(current_group)

    staff_measure_counts = [len(g) for g in staff_groups]
    n_staves = len(staff_groups)

    avg_start_x = np.mean([
        min([b[0] - b[2] / 2 for b in g]) for g in staff_groups
    ])

    staff_y_centers = [
        np.mean([b[1] for b in g]) for g in staff_groups
    ]

    return (
        n_staves,
        staff_measure_counts,
        staff_height,
        avg_start_x,
        staff_y_centers,
        target_label_boxes_with_conf
    )


# Example usage
if __name__ == "__main__":
    yolo_txt_path = "example_box/page_001.txt"
    img_path = "example_img/page_001.png"
    n_staves, measure_counts, staff_height, avg_start_x, staff_y_centers, boxes_with_conf = \
        count_staves_from_yolo(yolo_txt_path, img_path)

    print("Number of staves:", n_staves)
    print("Measures per staff:", measure_counts)
    print("Staff height (normalized):", staff_height)
    print("Average start x (normalized):", avg_start_x)
    print("Staff y centers (normalized):", staff_y_centers)
    print("Target label YOLO boxes (xc, yc, w, h, conf):")
    for b in boxes_with_conf:
        print(b)
