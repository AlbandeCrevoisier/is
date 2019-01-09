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

    Complexity: Let m*n be the dimensions of the matrix.
    Since we need to visit every pixel, we know that the lower bounds
    on complexity and memory size are O(m*n) in both cases.
    This algorithm has two elemnts: on one hand, it visits every pixel
    and checks wether it is a '1' or not. If it is, it visits its entire
    cluster and appends it to the list of clusters. The complexity of
    the algorithm is thus something in (m*n)*(complexity of exploring
    the clusters). However, we notice that once a cluster has been
    visited, each of its pixel has been set to '0', thus not prompting
    further exploration. This means that a '0' is visited once and a '1'
    at most twice (once if it's the element by which the cluster is
    discovered, twice otherwise, as the '0' will be checked later on).
    The amortized complexity of exploring the clusters yields an overall
    complexity in O(m*n).
    As for memory, the matrix and the list of clusters is stored. At
    most, every pixel will be in a cluster, so the list of clusters can
    go up to m*n pixels. The memory cost is thus 2*m*n, or O(m*n).
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

def get_mask(d):
    """Get the pixels at distance d or less."""
    m = []
    for i in range(-d, d+1):
        for j in range(-d, d+1):
            if i*i + j*j <= d*d:
                m.append([i, j])
    return m

