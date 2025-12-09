import cv2
import numpy as np
from PIL import Image
import os

# Input and output paths
INPUT_PATH = "input/content.jpeg"            # your original image
OUTPUT_PATH = "output/cartoon_disneyish.png"  # cartoon-style output


def cartoonize_opencv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple cartoon / Disney-ish effect using OpenCV:
    - Smooth colors (like 3D toon shading)
    - Detect edges
    - Combine both
    """
    # 1. Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Smooth the gray image to remove noise
    gray_blur = cv2.medianBlur(gray, 7)

    # 3. Edge detection with cleaner results
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=7,  # was 9
        C=5,          # was 2 → increase to reduce noise
    )

    # Remove small noise via morphological operations
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: slight dilation to make outlines smoother & thicker
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 4. Color smoothing (bilateral filter keeps edges, smooths colors)
    color = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Reduce number of colors (gives that flat cartoon shading)
    data = np.float32(color).reshape((-1, 3))
    K = 6  # number of color clusters, smaller -> more flat shading
    compactness, labels, centers = cv2.kmeans(
        data,
        K,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        5,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(color.shape)

    # 6. Combine quantized colors with edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized, edges_colored)

    return cartoon


def main():
    # Ensure folders exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load input image with OpenCV (BGR)
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input image not found at {INPUT_PATH}")

    img_bgr = cv2.imread(INPUT_PATH)
    if img_bgr is None:
        raise ValueError("Failed to load image. Check path and format.")

    # Apply cartoon / Disney-ish effect
    cartoon_bgr = cartoonize_opencv(img_bgr)

    # Save output
    cv2.imwrite(OUTPUT_PATH, cartoon_bgr)
    print(f"Saved cartoon-style image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
