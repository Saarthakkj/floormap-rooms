import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pipeline import extract_rooms

SAMPLES_DIR = Path(__file__).parent / "samples"

st.set_page_config(page_title="Floor Map Room Extractor", layout="wide")
st.title("Floor Map Room Extractor")

with st.sidebar:
    st.header("Parameters")
    close_kernel = st.slider(
        "Door gap kernel size",
        min_value=3, max_value=50, value=15, step=2,
        help="Larger = bridges wider gaps (doors). Too large merges adjacent rooms.",
    )
    min_area = st.slider(
        "Minimum room area (px)",
        min_value=100, max_value=10000, value=3000, step=100,
        help="Regions smaller than this are ignored as noise.",
    )
    wall_include = st.slider(
        "Wall inclusion (px)",
        min_value=0, max_value=40, value=15, step=1,
        help="How many pixels of surrounding wall to include in each room crop.",
    )
    run_ocr = st.checkbox("Extract room labels (OCR)", value=True)

uploaded = st.file_uploader(
    "Upload a floor plan image", type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

samples = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))
if samples and uploaded is None:
    if st.button(f"Load sample: {samples[0].name}"):
        st.session_state["sample"] = str(samples[0])
        st.rerun()

source = uploaded
if source is None and "sample" in st.session_state:
    source = st.session_state["sample"]

if source is not None:
    if isinstance(source, str):
        pil_img = Image.open(source).convert("RGB")
    else:
        pil_img = Image.open(source).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting rooms..."):
        annotated_bgr, rooms = extract_rooms(
            img_bgr,
            close_kernel_size=close_kernel,
            min_room_area=min_area,
            wall_include=wall_include,
            run_ocr=run_ocr,
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader(f"Detected: {len(rooms)} rooms")
        st.image(annotated_rgb, use_container_width=True)

    if rooms:
        st.subheader("Individual Room Crops")

        cols_per_row = 4
        for row_start in range(0, len(rooms), cols_per_row):
            row_rooms = rooms[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, room in zip(cols, row_rooms):
                crop_rgba = cv2.cvtColor(room["crop"], cv2.COLOR_BGRA2RGBA)
                col.image(crop_rgba, caption=room["name"], use_container_width=True)

        st.subheader("Room List")
        table_data = []
        for r in rooms:
            ch, cw = r["crop"].shape[:2]
            color_hex = "#{:02x}{:02x}{:02x}".format(*r["color"])
            table_data.append({
                "Name": r["name"],
                "Area (px)": r["area_px"],
                "Crop size": f"{cw}x{ch}",
                "Color": color_hex,
            })
        st.dataframe(table_data, use_container_width=True)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, r in enumerate(rooms):
                crop_rgba = cv2.cvtColor(r["crop"], cv2.COLOR_BGRA2RGBA)
                pil_crop = Image.fromarray(crop_rgba)
                img_buf = io.BytesIO()
                pil_crop.save(img_buf, format="PNG")
                safe_name = r["name"].replace(" ", "_").replace("/", "_").replace("#", "")
                zf.writestr(f"{i+1:02d}_{safe_name}.png", img_buf.getvalue())

        st.download_button(
            f"Download all {len(rooms)} room crops (.zip)",
            data=zip_buf.getvalue(),
            file_name="room_crops.zip",
            mime="application/zip",
        )
