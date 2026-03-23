import cv2
from scanner import get_document_contour
from utils import show_image, four_point_transform

image_path = "data/input/test.jpg"

image, contour = get_document_contour(image_path)

# check properly
if contour is None or len(contour) == 0:
    print("No document detected")
else:
    # convert contour to int (IMPORTANT)
    contour = contour.astype(int)

    # draw contour
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    show_image("Detected Document", image)

    # perspective transform
    warped = four_point_transform(image, contour)
    show_image("Warped", warped)

    # grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # threshold
    scan = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    show_image("Scanned (Final)", scan)

    # save
    cv2.imwrite("data/output/scanned.jpg", scan)
