import cv2
import numpy as np
from src.utils import resize_image


def get_document_contour(image_path):
    orig = cv2.imread(image_path)
    if orig is None:
        print("Image not found")
        return None, None

    image = resize_image(orig)
    ratio = orig.shape[1] / float(image.shape[1])

    h, w = image.shape[:2]

    # ---- Step 1: GrabCut (separate foreground) ----
    mask = np.zeros(image.shape[:2], np.uint8)

    rect = (20, 20, w - 40, h - 40)  # assume paper is inside
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmented = image * mask2[:, :, np.newaxis]

    # ---- Step 2: Edge detection ----
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # strengthen edges
    kernel = np.ones((5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # ---- Step 3: Find contours ----
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            document_contour = approx
            break

    # fallback
    if document_contour is None and len(contours) > 0:
        cnt = contours[0]
        peri = cv2.arcLength(cnt, True)
        document_contour = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    # scale back to original
    if document_contour is not None:
        document_contour = document_contour.astype("float32") * ratio

    return orig, document_contour
