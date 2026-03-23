import cv2
import numpy as np
from utils import resize_image


def get_document_contour(image_path):
    # read image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return None, None

    orig = image.copy()

    # resize for faster processing
    image = resize_image(image)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge detection
    edges = cv2.Canny(blur, 50, 150)

    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    for cnt in contours:
        # approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # if 4 points → likely document
        if len(approx) == 4:
            document_contour = approx
            break

    return image, document_contour
