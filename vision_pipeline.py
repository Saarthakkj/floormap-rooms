import base64
import json
import os
import cv2
import numpy as np
from openai import AzureOpenAI

from pipeline import extract_rooms, COLORS


def _encode_image(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


LABEL_PROMPT = """You are an architectural floor plan analyzer. I will show you:
1. The original floor plan image
2. An annotated version where each detected room is highlighted with a unique color and labeled with a temporary ID like "Room 1", "Room 2", etc.

Your job: read the text labels in the ORIGINAL floor plan image and match them to the correct Room IDs from the annotated image.

For each room, find its real name from the floor plan text (e.g. "KING #1 202", "CORRIDOR A", "LINEN", "STAIR A", "ELEV. 1", "ELEV. LOBBY", "QQ #2 204").

Return ONLY valid JSON:
{"labels": {"Room 1": "ACTUAL NAME", "Room 2": "ACTUAL NAME", ...}}

If you cannot determine a room's real name, keep its temporary ID."""


def _get_room_labels(original: np.ndarray, annotated: np.ndarray, room_ids: list[str]) -> dict[str, str]:
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://saarthak-openai.openai.azure.com/",
        api_key=os.environ.get("AZURE_OPENAI_KEY", ""),
    )

    orig_b64 = _encode_image(original)
    ann_b64 = _encode_image(annotated)

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": LABEL_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Here are the room IDs to label: {room_ids}\n\nFirst image: original floor plan. Second image: annotated version with room IDs."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ann_b64}"}},
            ]},
        ],
        temperature=0.1,
        max_completion_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    return data.get("labels", {})


def extract_rooms_vision(
    img: np.ndarray,
    close_kernel_size: int = 15,
    min_room_area: int = 3000,
    wall_include: int = 15,
) -> tuple[np.ndarray, list[dict]]:
    """Classical CV for boundaries + Vision API for labeling."""

    # Step 1: Run classical pipeline WITHOUT OCR to get room boundaries
    annotated_temp, rooms = extract_rooms(
        img,
        close_kernel_size=close_kernel_size,
        min_room_area=min_room_area,
        wall_include=wall_include,
        run_ocr=False,
    )
    print(f"[vision] Classical pipeline found {len(rooms)} rooms")

    if not rooms:
        return annotated_temp, rooms

    # Step 2: Build a clear annotated image with Room IDs for the vision model
    labeled_img = img.copy()
    overlay = img.copy()
    room_ids = []

    for idx, room in enumerate(rooms):
        room_id = f"Room {idx + 1}"
        room_ids.append(room_id)
        room["name"] = room_id

        color = room["color"]
        cx, cy = room["centroid"]

        # Draw large, clear label on the annotated image for the vision model
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        (tw, th), _ = cv2.getTextSize(room_id, font, scale, thickness)
        tx, ty = cx - tw // 2, cy + th // 2

        cv2.rectangle(labeled_img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(labeled_img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), color, 2)
        cv2.putText(labeled_img, room_id, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Apply color overlay from the classical pipeline
    annotated_for_api = cv2.addWeighted(annotated_temp, 0.6, labeled_img, 0.4, 0)
    # Actually, let's make a cleaner version: original + color overlay + labels
    for idx, room in enumerate(rooms):
        color = room["color"]
        # We need the mask — reconstruct from the pipeline
        # The annotated_temp already has the overlay, just add our labels on top
    annotated_for_api = annotated_temp.copy()
    for idx, room in enumerate(rooms):
        room_id = f"Room {idx + 1}"
        color = room["color"]
        cx, cy = room["centroid"]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        (tw, th), _ = cv2.getTextSize(room_id, font, scale, thickness)
        tx, ty = cx - tw // 2, cy + th // 2

        cv2.rectangle(annotated_for_api, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(annotated_for_api, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), color, 2)
        cv2.putText(annotated_for_api, room_id, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Step 3: Send both images to vision API for labeling
    print(f"[vision] Sending to API for labeling {len(room_ids)} rooms...")
    label_map = _get_room_labels(img, annotated_for_api, room_ids)
    print(f"[vision] Labels received: {label_map}")

    # Step 4: Apply labels to rooms and rebuild annotated image
    for idx, room in enumerate(rooms):
        room_id = f"Room {idx + 1}"
        real_name = label_map.get(room_id, room_id)
        room["name"] = real_name

    # Rebuild the annotated image with real names
    annotated_final = annotated_temp.copy()
    for room in rooms:
        cx, cy = room["centroid"]
        name = room["name"]
        cv2.putText(
            annotated_final, name, (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated_final, name, (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated_final, rooms
