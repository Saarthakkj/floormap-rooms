import base64
import io
import json
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler


def process_image(img_bytes, close_kernel_size=15, min_room_area=3000, wall_include=15):
    print(f"[extract] importing cv2...")
    import cv2
    import numpy as np
    from PIL import Image
    print(f"[extract] cv2 imported, decoding image ({len(img_bytes)} bytes)...")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("[extract] ERROR: cv2.imdecode returned None")
        return None

    print(f"[extract] image decoded: {img.shape}, importing vision pipeline...")
    from vision_pipeline import extract_rooms_vision

    print(f"[extract] running extract_rooms_vision(kernel={close_kernel_size}, area={min_room_area}, wall={wall_include})...")
    annotated_bgr, rooms = extract_rooms_vision(
        img,
        close_kernel_size=close_kernel_size,
        min_room_area=min_room_area,
        wall_include=wall_include,
    )
    print(f"[extract] done, {len(rooms)} rooms found, encoding response...")

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

    print(f"[extract] response ready, {len(room_list)} crops encoded")
    return {"annotated": annotated_b64, "rooms": room_list}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            print("[extract] POST received")
            content_length = int(self.headers.get("Content-Length", 0))
            print(f"[extract] Content-Length: {content_length}")

            if content_length == 0 or content_length > 10 * 1024 * 1024:
                self._json_error(400, "Invalid image size")
                return

            body = self.rfile.read(content_length)
            print(f"[extract] body read: {len(body)} bytes")

            content_type = self.headers.get("Content-Type", "")
            img_bytes = None
            params = {}

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[1].split(";")[0].encode()
                parts = body.split(b"--" + boundary)
                print(f"[extract] multipart: {len(parts)} parts, boundary={boundary[:20]}")
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
                        print(f"[extract] found image part: {len(img_bytes)} bytes")
                    elif 'name="close_kernel_size"' in header_str:
                        params["close_kernel_size"] = int(part_body.strip())
                    elif 'name="min_room_area"' in header_str:
                        params["min_room_area"] = int(part_body.strip())
                    elif 'name="wall_include"' in header_str:
                        params["wall_include"] = int(part_body.strip())
            else:
                img_bytes = body
                print(f"[extract] raw body as image: {len(img_bytes)} bytes")

            if img_bytes is None:
                print("[extract] ERROR: no image found in request")
                self._json_error(400, "No image provided")
                return

            result = process_image(img_bytes, **params)

            if result is None:
                self._json_error(400, "Failed to decode image")
                return

            response = json.dumps(result)
            print(f"[extract] sending response: {len(response)} bytes")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response.encode())

        except Exception as e:
            print(f"[extract] EXCEPTION: {e}")
            traceback.print_exc()
            self._json_error(500, str(e))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_error(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode())
