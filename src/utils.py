import cv2
import numpy as np


def resize_image(image, width=800):
    h, w = image.shape[:2]
    ratio = width / float(w)
    new_dim = (width, int(h * ratio))
    return cv2.resize(image, new_dim)


def order_points(pts):
    pts = pts.reshape(-1, 2)

    # if more than 4 points, pick 4 extreme ones
    if len(pts) > 4:
        # sort by x + y (top-left, bottom-right)
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        # sort by difference (top-right, bottom-left)
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        rect = np.array([tl, tr, br, bl], dtype="float32")
    else:
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

    return rect

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def four_point_transform(image, pts):
    #from utils import order_points

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # compute height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
