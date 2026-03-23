import cv2
from scanner import get_document_contour
from utils import show_image, four_point_transform

image_path = "data/input/test.jpg"

image, contour = get_document_contour(image_path)

if contour is None or len(contour) == 0:
    print("No document detected")
else:
    contour = contour.astype(int)

    # draw contour
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    show_image("Detected Document", image)

    # perspective transform
    warped = four_point_transform(image, contour)
    show_image("Warped", warped)

    # ===== FINAL CLEAN SCAN PIPELINE =====

    # grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # ---- STEP 1: remove shadows ----
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized = cv2.divide(gray, bg, scale=255)

    # ---- STEP 2: denoise ----
    denoised = cv2.fastNlMeansDenoising(normalized, None, 30, 7, 21)

    # ---- STEP 3: threshold ----
    _, scan = cv2.threshold(denoised, 140, 255, cv2.THRESH_BINARY)

    # show result
    show_image("Scanned (Final)", scan)

    # save output
    cv2.imwrite("data/output/scanned.jpg", scan)
