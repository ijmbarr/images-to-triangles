import numpy as np
from skimage import filters, morphology, color
from triangulared.utils import gaussian_mask, default


def edge_points(image, length_scale=200,
                n_horizontal_points=None,
                n_vertical_points=None):
    """
    Returns points around the edge of an image.
    :param image: image array
    :param length_scale: how far to space out the points if no
                         fixed number of points is given
    :param n_horizontal_points: number of points on the horizonal edge.
                                Leave as None to use lengthscale to determine
                                the value
    :param n_vertical_points: number of points on the horizonal edge.
                                Leave as None to use lengthscale to determine
                                the value
    :return: array of coordinates
    """
    ymax, xmax = image.shape[:2]

    if n_horizontal_points is None:
        n_horizontal_points = int(xmax / length_scale)

    if n_vertical_points is None:
        n_vertical_points = int(ymax / length_scale)

    delta_x = xmax / n_horizontal_points
    delta_y = ymax / n_vertical_points

    return np.array(
        [[0, 0], [xmax, 0], [0, ymax], [xmax, ymax]]
        + [[delta_x * i, 0] for i in range(1, n_horizontal_points)]
        + [[delta_x * i, ymax] for i in range(1, n_horizontal_points)]
        + [[0, delta_y * i] for i in range(1, n_vertical_points)]
        + [[xmax, delta_y * i] for i in range(1, n_vertical_points)]
    )


def generate_uniform_random_points(image, n_points=100):
    """
    Generates a set of uniformly distributed points over the area of image
    :param image: image as an array
    :param n_points: int number of points to generate
    :return: array of points
    """
    ymax, xmax = image.shape[:2]
    points = np.random.uniform(size=(n_points, 2))
    points *= np.array([xmax, ymax])
    points = np.concatenate([points, edge_points(image)])
    return points


def generate_max_entropy_points(image, n_points=100,
                                entropy_width=None,
                                filter_width=None,
                                suppression_width=None,
                                suppression_amplitude=None):
    """
    Generates a set of points over the area of image, using maximum entropy
    to guess which points are importance. All length scales are relative to the
    density of the points.
    :param image: image as an array
    :param n_points: int number of points to generate:
    :param entropy_width: width over which to measure entropy
    :param filter_width: width over which to pre filter entropy
    :param suppression_width: length for suppressing entropy before choosing the
                              next point.
    :param suppression_amplitude: amplitude to suppress entropy before choosing the
                              next point.
    :return:
    """
    # calculate length scale
    ymax, xmax = image.shape[:2]
    length_scale = np.sqrt(xmax*ymax / n_points)
    entropy_width = length_scale * default(entropy_width, 0.2)
    filter_width = length_scale * default(filter_width, 0.1)
    suppression_width = length_scale * default(suppression_width, 0.3)
    suppression_amplitude = default(suppression_amplitude, 3)

    # convert to grayscale
    im2 = color.rgb2gray(image)

    # filter
    im2 = (
        255 * filters.gaussian(im2, sigma=filter_width, multichannel=True)
    ).astype("uint8")

    # calculate entropy
    im2 = filters.rank.entropy(im2, morphology.disk(entropy_width))

    points = []
    for _ in range(n_points):
        y, x = np.unravel_index(np.argmax(im2), im2.shape)
        im2 -= gaussian_mask(x, y,
                             shape=im2.shape[:2],
                             amp=suppression_amplitude,
                             sigma=suppression_width)
        points.append((x, y))

    points = np.array(points)
    return points
