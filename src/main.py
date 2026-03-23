import cv2
from scanner import get_document_contour
from utils import show_image

image, contour = get_document_contour("data/input/test.jpg")

if contour is None:
    print("No document detected")
else:
    # draw contour
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    show_image("Detected Document", image)
