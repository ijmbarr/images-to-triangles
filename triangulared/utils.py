import pandas as pd
import numpy as np


def get_triangle_colour(triangles, image, agg_func=np.median):
    """
    Get's the colour of a triangle, based on applying agg_func to the pixels
    under it
    :param triangles: scipy.spatial.Delaunay
    :param image: image as array
    :param agg_func: function
    :return: colour list
    """
    # create a list of all pixel coordinates
    ymax, xmax = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(xmax), np.arange(ymax))
    pixel_coords = np.c_[xx.ravel(), yy.ravel()]

    # for each pixel, identify which triangle it belongs to
    triangles_for_coord = triangles.find_simplex(pixel_coords)

    df = pd.DataFrame({
        "triangle": triangles_for_coord,
        "r": image.reshape(-1, 3)[:, 0],
        "g": image.reshape(-1, 3)[:, 1],
        "b": image.reshape(-1, 3)[:, 2]
    })

    n_triangles = triangles.vertices.shape[0]

    by_triangle = (
        df
            .groupby("triangle")
        [["r", "g", "b"]]
            .aggregate(agg_func)
            .reindex(range(n_triangles), fill_value=0)
        # some triangles might not have pixels in them
    )

    return by_triangle.values / 256


def gaussian_mask(x, y, shape, amp=1, sigma=15):
    """
    Returns an array of shape, with values based on

    amp * exp(-((i-x)**2 +(j-y)**2) / (2 * sigma ** 2))

    :param x: float
    :param y: float
    :param shape: tuple
    :param amp: float
    :param sigma: float
    :return: array
    """
    xv, yv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    g = amp * np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
    return g


def default(value, default_value):
    """
    Returns default_value if value is None, value otherwise
    """
    if value is None:
        return default_value
    return value
