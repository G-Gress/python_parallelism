"""
Preprocessing functions for images.
"""

from typing import List, Any
from PIL import Image
import numpy as np

def load_resize(path: Any, size: tuple = (224,224)) -> np.array:
    """
    Open an image from a path, convert it to RGB to ensure we have 3 channels, and resize it.
    Returns a NumPy array of shape (height, width, 3).

    Args:
        path (Any): The path to the image file.
        size (tuple, optional): The target size for resizing. Defaults to (224,224).

    Returns:
        np.array: The preprocessed image as a NumPy array.
    """

    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

def to_gray(img_rgb: np.array) -> np.array:
    """
    Convert an RGB image of shape (height, width, 3) to grayscale using perceptual weights.
    Returns a NumPy array of shape (height, width).

    Args:
        img_rgb (np.array): The input RGB image.

    Returns:
        np.array: The grayscale image.
    """

    # Use ellipsis (...) to select all pixels for each channel
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    # Apply ITU-R 601 to calculate luminance
    # Cast float32 to avoid overflow later
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

def sobel_edges(img_gray: np.array) -> np.array:
    """
    Apply the Sobel operator to a grayscale image of shape (height, width) to detect edges.
    Returns a NumPy array of the same shape as the input image.

    Args:
        img_gray (np.array): The input grayscale image.

    Returns:
        np.array: The edge-detected image.
    """
    # Create two 3x3 Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[ 1, 2, 1],
                        [ 0, 0, 0],
                        [-1,-2,-1]], dtype=np.float32)

    # Pad the image to handle borders
    grad = np.pad(img_gray, ((1, 1), (1, 1)), mode='edge')
    # Initialize gradient images
    grad_x = (
        grad[:-2, :-2] * sobel_x[0, 0] + grad[:-2, 1:-1] * sobel_x[0, 1] + grad[:-2, 2:] * sobel_x[0, 2] +
        grad[1:-1, :-2] * sobel_x[1, 0] + grad[1:-1, 1:-1] * sobel_x[1, 1] + grad[1:-1, 2:] * sobel_x[1, 2] +
        grad[2:, :-2] * sobel_x[2, 0] + grad[2:, 1:-1] * sobel_x[2, 1] + grad[2:, 2:] * sobel_x[2, 2]
    )
    grad_y = (
        grad[:-2, :-2] * sobel_y[0, 0] + grad[:-2, 1:-1] * sobel_y[0, 1] + grad[:-2, 2:] * sobel_y[0, 2] +
        grad[1:-1, :-2] * sobel_y[1, 0] + grad[1:-1, 1:-1] * sobel_y[1, 1] + grad[1:-1, 2:] * sobel_y[1, 2] +
        grad[2:, :-2] * sobel_y[2, 0] + grad[2:, 1:-1] * sobel_y[2, 1] + grad[2:, 2:] * sobel_y[2, 2]
    )
    # Compute and return the gradient magnitude
    return np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

def preprocess_one(path: Any, size: tuple = (224, 224), bins: int = 32) -> dict:
    """
    Preprocess pipeline for one image.
    Returns a dictionary containing components for reuse.

    Args:
        path (Any): The path to the image file.
        size (tuple, optional): The target size for resizing. Defaults to (224, 224).
        bins (int, optional): The number of bins for color histograms. Defaults to 32.

    Returns:
        dict: A dictionary containing the preprocessed image components.
    """

    img_resized = load_resize(path, size)
    img_gray = to_gray(img_resized)
    img_edges = sobel_edges(img_gray)

    hists = []
    # Compute color histograms for each channel
    for channel in range(3):
        hist_channel, _ = np.histogram(img_resized[..., channel], bins=bins, range=(0, 255))
        hists.append(hist_channel.astype(np.float32))

    color_hist = np.concatenate(hists, axis=0)

    return {
        "path": str(path),
        "resized": img_resized,
        "gray": img_gray,
        "edges": img_edges,
        "color_hist": color_hist,
        "size": size,
        "bins": bins
    }

def preprocess_one_min(path: Any, size: tuple = (224, 224), bins: int = 32) -> dict:
    """
    Preprocess pipeline for one image.
    Returns a minimal dictionary of features.

    Args:
        path (Any): The path to the image file.
        size (tuple, optional): The target size for resizing. Defaults to (224, 224).
        bins (int, optional): The number of bins for color histograms. Defaults to 32.

    Returns:
        dict: A minimal dictionary of features.
    """
    img_resized = load_resize(path, size)
    img_gray = to_gray(img_resized)
    img_edges = sobel_edges(img_gray)
    # Simple feature: edge mean + color hist (density)
    hists = []
    # Compute color histograms for each channel
    for channel in range(3):
        hist_channel, _ = np.histogram(img_resized[..., channel], bins=bins, range=(0,255), density=True)
        hists.append(hist_channel.astype(np.float32))
    color_hist = np.concatenate(hists)

    return {
        "path": str(path),
        "edge_mean": float(img_edges.mean()),
        "edge_max": float(img_edges.max()),
        "color_hist": color_hist
    }

def preprocess_all_serial(image_paths: List[str], size: tuple = (224, 224), bins: int = 32, limit: int = None) -> List[dict]:
    """
    Preprocess pipeline for a list of images.
    Returns a list of dictionaries, each containing components for reuse.

    Args:
        image_paths (List): A list of paths to the image files.
        size (tuple, optional): The target size for resizing. Defaults to (224, 224).
        bins (int, optional): The number of bins for color histograms. Defaults to 32.
        limit (int, optional): The maximum number of images to process. Defaults to None.

    Returns:
        list: A list of dictionaries, each containing components for reuse.
    """
    if limit is not None:
        image_paths = image_paths[:limit]
    return [preprocess_one(path, size=size, bins=bins) for path in image_paths]
