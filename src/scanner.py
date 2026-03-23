import cv2
import numpy as np
from utils import resize_image


def get_document_contour(image_path):
    # read original image
    orig = cv2.imread(image_path)
    if orig is None:
        print("Image not found")
        return None, None

    image = resize_image(orig)

    # compute scale ratio
    ratio = orig.shape[1] / float(image.shape[1])

    # preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge detection
    edges = cv2.Canny(blur, 50, 150)

    # dilate edges (connect broken lines)
    kernel = np.ones((5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 1500:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            document_contour = approx
            break

    # fallback: use largest contour
    if document_contour is None and len(contours) > 0:
        cnt = contours[0]
        peri = cv2.arcLength(cnt, True)
        document_contour = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    # scale contour back to original image size
    if document_contour is not None:
        document_contour = document_contour.astype("float32") * ratio

    return orig, document_contour
