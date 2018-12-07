"""Internship offline test: cluster detection in a binary image.

Convention: the origin of (x, y) coordinates is the top left corner of
the image, with x increasing towards the right and y increasing towards
the bottom.

Assumption: in PART 1: Simple Clustering, it is said that '2 pixels of
ones are in the same cluster, if they touch each other'. I will assume
that this does not take into account the diagonals.
"""


def clustering(image, cluster):
    """Find clusters in an image and register them.

    cluster: function that registers the cluster found at (x, y).
    """
    im = image.copy()
    clusters = []
    for y in range(len(im)):
        for x in range(len(im[y])):
            if im[y][x] is 1:
                clusters.append([])
                cluster(im, x, y, clusters)
    return clusters


def simple_cluster(image, x, y, clusters):
    """Register the simple cluster around (x, y) in image.

    Replace the '1's of the cluster around (x, y) by '0's and add the
    corresponding pixel coordinates to the cluster.
    """
    clusters[-1].append([x, y])
    image[y][x] = 0
    if y is not 0 and image[y-1][x] is 1:
        simple_cluster(image, x, y-1, clusters)
    if x is not 0 and image[y][x-1] is 1:
        simple_cluster(image, x-1, y, clusters)
    if x is not len(image[0])-1 and image[y][x+1] is 1:
        simple_cluster(image, x+1, y, clusters)
    if y is not len(image)-1 and image[y+1][x] is 1:
        simple_cluster(image, x, y+1, clusters)


def d_cluster(image, x, y, d, clusters):
    """Register the cluster around (x, y) in image.

    d is the static distance that defines a cluster.
    """
        clusters[-1].append([x, y])
    image[y][x] = 0
