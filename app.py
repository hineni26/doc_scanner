import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.scanner import get_document_contour
from src.utils import four_point_transform

st.title("Document Scanner")

uploaded_files = st.file_uploader(
    "Upload images", accept_multiple_files=True, type=["jpg", "png", "jpeg"]
)

scanned_images = []

if uploaded_files:
    for file in uploaded_files:
        # read image
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # save temp
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, image)

        # detect contour
        img, contour = get_document_contour(temp_path)

        if contour is not None:
            contour = contour.astype(int)

            warped = four_point_transform(img, contour)

            # ===== CLEAN PIPELINE =====
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            bg = cv2.GaussianBlur(gray, (51, 51), 0)
            normalized = cv2.divide(gray, bg, scale=255)

            denoised = cv2.fastNlMeansDenoising(normalized, None, 30, 7, 21)

            _, scan = cv2.threshold(denoised, 140, 255, cv2.THRESH_BINARY)

            st.image(scan, caption="Scanned Page", use_column_width=True)

            scanned_images.append(Image.fromarray(scan))
        else:
            st.warning("Document not detected in one image")

    # ===== PDF DOWNLOAD =====
    if scanned_images:
        pdf_path = "output.pdf"
        scanned_images[0].save(
            pdf_path,
            save_all=True,
            append_images=scanned_images[1:]
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="scanned_document.pdf"
            )
