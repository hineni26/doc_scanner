import cv2
from utils import resize_image, show_image

img = cv2.imread("data/input/test.jpg")

if img is None:
    print("Image not found")
else:
    img = resize_image(img)
    show_image("Test", img)
