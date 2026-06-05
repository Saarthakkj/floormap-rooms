import cv2
import numpy as np


COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def extract_rooms(
    img: np.ndarray,
    close_kernel_size: int = 15,
    min_room_area: int = 3000,
    wall_include: int = 15,
    run_ocr: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    kernel_door = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel_door, iterations=1)
    room_mask = cv2.bitwise_not(walls_closed)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        room_mask, connectivity=4
    )

    img_h, img_w = room_mask.shape
    areas = stats[:, cv2.CC_STAT_AREA]

    valid = []
    for i in range(1, num_labels):
        if areas[i] < min_room_area:
            continue
        if _touches_border(stats[i], img_w, img_h):
            continue
        valid.append(i)

    text_map = {}
    if run_ocr and valid:
        text_map = _run_ocr(img, labels, valid)

    adj_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_wall = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (wall_include * 2 + 1, wall_include * 2 + 1)
    )

    annotated = img.copy()
    overlay = img.copy()
    rooms = []

    for idx, label_id in enumerate(valid):
        color = COLORS[idx % len(COLORS)]
        mask = (labels == label_id).astype(np.uint8) * 255

        overlay[mask == 255] = color

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(annotated, [approx], -1, color, 2)

        full_mask = _merge_adjacent(
            mask, labels, num_labels, areas, areas[label_id], adj_kernel,
            min_room_area, valid_labels=valid, current_label=label_id,
        )

        crop_rgba = _crop_room_to_walls(
            img, full_mask, binary, kernel_wall, img_w, img_h
        )

        name = text_map.get(label_id, f"Room {idx + 1}")
        cx, cy_ = int(centroids[label_id][0]), int(centroids[label_id][1])

        rooms.append({
            "name": name,
            "area_px": int(areas[label_id]),
            "centroid": (cx, cy_),
            "color": color,
            "crop": crop_rgba,
        })

    alpha = 0.35
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

    for room in rooms:
        cx, cy_ = room["centroid"]
        cv2.putText(
            annotated, room["name"], (cx - 30, cy_),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, room["name"], (cx - 30, cy_),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated, rooms


def _merge_adjacent(
    room_mask: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    areas: np.ndarray,
    room_area: int,
    adj_kernel: np.ndarray,
    min_room_area: int = 3000,
    valid_labels: list[int] | None = None,
    current_label: int | None = None,
) -> np.ndarray:
    dilated = cv2.dilate(room_mask, adj_kernel, iterations=1)
    merged = room_mask.copy()

    max_merge_area = min_room_area
    valid_set = set(valid_labels) if valid_labels else set()

    for i in range(1, num_labels):
        if areas[i] >= max_merge_area:
            continue
        other = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(dilated, other)
        if not np.any(overlap):
            continue
        if valid_set:
            other_dilated = cv2.dilate(other, adj_kernel, iterations=1)
            neighbor_count = 0
            for v in valid_set:
                if v == current_label or v == i:
                    continue
                vmask = (labels == v).astype(np.uint8) * 255
                if np.any(cv2.bitwise_and(other_dilated, vmask)):
                    neighbor_count += 1
                    if neighbor_count >= 1:
                        break
            if neighbor_count >= 1:
                continue
        merged = cv2.bitwise_or(merged, other)

    return merged


def _crop_room_to_walls(
    img: np.ndarray,
    room_mask: np.ndarray,
    wall_binary: np.ndarray,
    kernel_wall: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    dilated = cv2.dilate(room_mask, kernel_wall, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        x, y, w, h = cv2.boundingRect(room_mask)
        crop_bgr = img[y : y + h, x : x + w].copy()
        alpha_ch = np.full((h, w), 255, dtype=np.uint8)
        return np.dstack([crop_bgr, alpha_ch])

    filled = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    snap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    walls_near = cv2.bitwise_and(
        cv2.dilate(filled, snap_kernel), wall_binary
    )
    filled = cv2.bitwise_or(filled, walls_near)

    contours2, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(final, contours2, -1, 255, cv2.FILLED)

    x, y, w, h = cv2.boundingRect(final)
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    crop_bgr = img[y : y + h, x : x + w].copy()
    alpha_ch = final[y : y + h, x : x + w]

    b, g, r = cv2.split(crop_bgr)
    return cv2.merge([b, g, r, alpha_ch])


def _run_ocr(
    img: np.ndarray, labels: np.ndarray, valid_labels: list[int]
) -> dict[int, str]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img)

    valid_set = set(valid_labels)
    label_texts: dict[int, list[tuple[float, str, float]]] = {v: [] for v in valid_labels}

    for bbox, text, conf in results:
        if conf < 0.3 or len(text.strip()) < 2:
            continue
        lbl = _find_room_for_bbox(labels, bbox, valid_set)
        if lbl is not None:
            y_center = sum(p[1] for p in bbox) / len(bbox)
            label_texts[lbl].append((conf, text.strip(), y_center))

    text_map = {}
    for lbl, texts in label_texts.items():
        if not texts:
            continue
        alpha_texts = [(c, t, y) for c, t, y in texts if any(ch.isalpha() for ch in t)]
        if alpha_texts:
            alpha_texts.sort(key=lambda t: (-sum(ch.isalpha() for ch in t[1]), t[2]))
            text_map[lbl] = alpha_texts[0][1]
        else:
            texts.sort(key=lambda t: (-len(t[1]), -t[0]))
            text_map[lbl] = texts[0][1]

    return text_map


def _touches_border(stat_row: np.ndarray, img_w: int, img_h: int, margin: int = 2) -> bool:
    x = stat_row[cv2.CC_STAT_LEFT]
    y = stat_row[cv2.CC_STAT_TOP]
    w = stat_row[cv2.CC_STAT_WIDTH]
    h = stat_row[cv2.CC_STAT_HEIGHT]
    return x <= margin or y <= margin or (x + w) >= img_w - margin or (y + h) >= img_h - margin


def _find_room_for_bbox(
    labels: np.ndarray, bbox: list, valid_set: set[int], expand: int = 10
) -> int | None:
    h, w = labels.shape
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    x_lo = max(0, min(xs) - expand)
    x_hi = min(w, max(xs) + expand + 1)
    y_lo = max(0, min(ys) - expand)
    y_hi = min(h, max(ys) + expand + 1)

    patch = labels[y_lo:y_hi, x_lo:x_hi].ravel()
    counts: dict[int, int] = {}
    for v in patch:
        if v in valid_set:
            counts[v] = counts.get(v, 0) + 1
    if counts:
        return max(counts, key=counts.get)
    return None
