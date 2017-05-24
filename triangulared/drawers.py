"""
Utilities to draw different steps to a matplotlib ax
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def set_axis_defaults(ax):
    """
    Set's some defaults for a matplotlib ax

    :param ax: ax to change
    :return: None
    """
    ax.axis("off")
    ax.axis("tight")
    ax.set_aspect("equal")
    ax.autoscale(False)


def draw_image(ax, image):
    """
    Plots image to an ax
    :param ax: matplotlib axis
    :param image: image in array form
    :return: None
    """
    ax.imshow(image)


def draw_points(ax, points):
    """
    Plots a set of points on an ax
    :param ax: ax
    :param points: array of (x,y) coordinates
    :return: None
    """
    ax.scatter(x=points[:, 0], y=points[:, 1], color="k")


def draw_triangles(ax, points, vertices, colours=None, **kwargs):
    """
    Draws a set of triangles on axis
    :param ax: ax
    :param points: array of (x,y) coordinates
    :param vertices: an array of the vertices of the triangles, indexing the array points
    :param colours: colour of the faces, set as none just to plot the outline
    :param kwargs: kwargs passed to Polygon
    :return: None
    """

    if colours is None:
        face_colours = len(vertices) * ["none"]
        line_colours = len(vertices) * ["black"]
    else:
        face_colours = colours
        line_colours = colours

    for triangle, fc, ec in zip(vertices, face_colours, line_colours):
        p = Polygon([points[i]
                     for i in triangle],
                    closed=True, facecolor=fc,
                    edgecolor=ec, **kwargs)
        ax.add_patch(p)
