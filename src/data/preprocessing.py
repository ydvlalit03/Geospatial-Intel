"""Image preprocessing utilities for satellite imagery."""

from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a GeoTIFF file and return image array + metadata.

    Args:
        path: Path to GeoTIFF file.

    Returns:
        Tuple of (image array [C, H, W], metadata dict).
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF reading")

    with rasterio.open(str(path)) as src:
        image = src.read()  # (C, H, W)
        meta = {
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform) if src.transform else None,
            "bounds": list(src.bounds) if src.bounds else None,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
        }
    return image, meta


def tile_image(
    image: np.ndarray, tile_size: int = 256, overlap: int = 0
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Split a large image into smaller tiles.

    Args:
        image: Image array of shape (C, H, W) or (H, W, C).
        tile_size: Size of each square tile.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        List of (tile_array, (row_offset, col_offset)) tuples.
    """
    if image.ndim == 3 and image.shape[0] <= image.shape[2]:
        # (C, H, W) format
        _, h, w = image.shape
        channel_first = True
    else:
        h, w = image.shape[0], image.shape[1]
        channel_first = False

    stride = tile_size - overlap
    tiles = []

    for row in range(0, h, stride):
        for col in range(0, w, stride):
            r_end = min(row + tile_size, h)
            c_end = min(col + tile_size, w)
            r_start = max(0, r_end - tile_size)
            c_start = max(0, c_end - tile_size)

            if channel_first:
                tile = image[:, r_start:r_end, c_start:c_end]
            else:
                tile = image[r_start:r_end, c_start:c_end]

            tiles.append((tile, (r_start, c_start)))

    return tiles


def normalize_image(
    image: np.ndarray, method: str = "minmax"
) -> np.ndarray:
    """Normalize image to [0, 1] range.

    Args:
        image: Input image array.
        method: 'minmax' for min-max scaling, 'standard' for z-score.

    Returns:
        Normalized image as float32.
    """
    image = image.astype(np.float32)

    if method == "minmax":
        vmin, vmax = image.min(), image.max()
        if vmax - vmin > 0:
            image = (image - vmin) / (vmax - vmin)
    elif method == "standard":
        mean, std = image.mean(), image.std()
        if std > 0:
            image = (image - mean) / std
    return image


def compute_ndvi(
    red: np.ndarray, nir: np.ndarray
) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Red band array.
        nir: Near-infrared band array.

    Returns:
        NDVI array with values in [-1, 1].
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denominator = nir + red
    ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
    return ndvi.astype(np.float32)
