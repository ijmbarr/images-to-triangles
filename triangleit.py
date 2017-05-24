from triangulared import generate_max_entropy_points, get_triangle_colour, draw_triangles, set_axis_defaults, edge_points
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import argparse


def process(input_path, output_path, n_points):
    image = plt.imread(input_path)
    points = generate_max_entropy_points(image, n_points=n_points)
    points = np.concatenate([points, edge_points(image)])

    tri = Delaunay(points)

    fig, ax = plt.subplots()
    ax.invert_yaxis()
    triangle_colours = get_triangle_colour(tri, image)
    draw_triangles(ax, tri.points, tri.vertices, triangle_colours)

    # remove boundary
    ax.axis("tight")
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ratio = image.shape[0] / image.shape[1]
    fig.set_size_inches(5, 5*ratio)

    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turns and image into triangles")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("-n", "--n_points", nargs='?',
                        help="number of points to use", default=100)

    ns = parser.parse_args()

    input_file = ns.input_file
    output_file = ns.output_file
    n_points = int(ns.n_points)

    process(input_path=input_file, output_path=output_file, n_points=n_points)



