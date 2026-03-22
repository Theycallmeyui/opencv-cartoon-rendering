import cv2
import numpy as np
import os

INPUT_IMAGE = "chainsawman.jpg"
OUTPUT_IMAGE = "cartoon_output.jpg"
EDGES_IMAGE = "edges_output.jpg"
COMPARISON_IMAGE = "comparison_output.jpg"

RESIZE_WIDTH = 800

MEDIAN_BLUR_KSIZE = 5
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

DILATION_KERNEL_SIZE = 2
DILATION_ITERATIONS = 1

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 250
BILATERAL_SIGMA_SPACE = 250


def resize_keep_ratio(img, target_width):
    h, w = img.shape[:2]

    if w <= target_width:
        return img

    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


def cartoonize(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, MEDIAN_BLUR_KSIZE)
    edges = cv2.Canny(gray_blur, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)
    color = cv2.bilateralFilter(
        img,
        BILATERAL_D,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE
    )
    edges_inv = cv2.bitwise_not(edges)
    cartoon = cv2.bitwise_and(color, color, mask=edges_inv)

    return gray, edges, cartoon


def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: '{INPUT_IMAGE}' not found.")
        print("Put your image in the same folder as this Python file.")
        return
    img = cv2.imread(INPUT_IMAGE)

    if img is None:
        print(f"Error: Could not open '{INPUT_IMAGE}'.")
        return
    img = resize_keep_ratio(img, RESIZE_WIDTH)
    gray, edges, cartoon = cartoonize(img)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((img, edges_bgr, cartoon))
    cv2.imwrite(OUTPUT_IMAGE, cartoon)
    cv2.imwrite(EDGES_IMAGE, edges)
    cv2.imwrite(COMPARISON_IMAGE, comparison)

    print(f"Saved cartoon image: {OUTPUT_IMAGE}")
    print(f"Saved edge image: {EDGES_IMAGE}")
    print(f"Saved comparison image: {COMPARISON_IMAGE}")
    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("Cartoon", cartoon)
    cv2.imshow("Comparison: Original | Edges | Cartoon", comparison)

    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()