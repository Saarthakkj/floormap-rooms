import base64
import io
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
from http.server import BaseHTTPRequestHandler


def process_image(img_bytes, close_kernel_size=15, min_room_area=3000, wall_include=15):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    from pipeline import extract_rooms

    annotated_bgr, rooms = extract_rooms(
        img,
        close_kernel_size=close_kernel_size,
        min_room_area=min_room_area,
        wall_include=wall_include,
        run_ocr=False,
    )

    _, ann_buf = cv2.imencode(".png", annotated_bgr)
    annotated_b64 = base64.b64encode(ann_buf.tobytes()).decode()

    room_list = []
    for r in rooms:
        crop_rgba = cv2.cvtColor(r["crop"], cv2.COLOR_BGRA2RGBA)
        pil_crop = Image.fromarray(crop_rgba)
        buf = io.BytesIO()
        pil_crop.save(buf, format="PNG")
        crop_b64 = base64.b64encode(buf.getvalue()).decode()

        room_list.append({
            "name": r["name"],
            "area_px": r["area_px"],
            "crop": crop_b64,
        })

    return {"annotated": annotated_b64, "rooms": room_list}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0 or content_length > 10 * 1024 * 1024:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid image size"}).encode())
            return

        body = self.rfile.read(content_length)

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            boundary = content_type.split("boundary=")[1].encode()
            parts = body.split(b"--" + boundary)
            img_bytes = None
            params = {}
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                header_body = part.split(b"\r\n\r\n", 1)
                if len(header_body) < 2:
                    continue
                header_str = header_body[0].decode(errors="ignore")
                part_body = header_body[1].rstrip(b"\r\n--")

                if 'name="image"' in header_str:
                    img_bytes = part_body
                elif 'name="close_kernel_size"' in header_str:
                    params["close_kernel_size"] = int(part_body.strip())
                elif 'name="min_room_area"' in header_str:
                    params["min_room_area"] = int(part_body.strip())
                elif 'name="wall_include"' in header_str:
                    params["wall_include"] = int(part_body.strip())
        else:
            img_bytes = body
            params = {}

        if img_bytes is None:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No image provided"}).encode())
            return

        result = process_image(img_bytes, **params)

        if result is None:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Failed to decode image"}).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
