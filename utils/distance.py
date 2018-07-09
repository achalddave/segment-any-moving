import random
from queue import PriorityQueue

import numpy as np


def chi_square_distance(histogram1, histogram2):
    """
    Chi-square distance between histograms.

    As in Sec 3.1 of:
        Belongie, Serge, Jitendra Malik, and Jan Puzicha. "Shape matching and
        object recognition using shape contexts." IEEE transactions on pattern
        analysis and machine intelligence 24.4 (2002): 509-522.
        <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.6897&rep=rep1&type=pdf>

    Args:
        histogram1 (array): Shape (num_bins, ). Normalized histogram.
        histogram2 (array): Shape (num_bins, ). Normalized histogram.

    Returns:
        distance (float)
    """
    histogram_sum = (histogram1 + histogram2)
    histogram_sum[histogram_sum == 0] = 1
    return 0.5 * ((
        (histogram1 - histogram2)**2) / (histogram_sum)).sum()


def intersection_distance(histogram1, histogram2):
    return np.max([histogram1, histogram2], axis=0).sum()


def histogram_distance(detection1, detection2):
    # Use chi-squared distance, as in
    #     Xiao, Jianxiong, et al. "Sun database: Large-scale scene recognition
    #     from abbey to zoo." Computer vision and pattern recognition (CVPR),
    #     2010 IEEE conference on. IEEE, 2010.
    # https://www.cc.gatech.edu/~hays/papers/sun.pdf
    #
    # See https://arxiv.org/pdf/1612.07408.pdf for references that analyze
    # chi-squared distance.
    # Extract detection.mask from detection.image
    # Compute histogram of colors over the mask
    histogram1, histogram1_edges = detection1.compute_histogram()
    histogram2, histogram2_edges = detection2.compute_histogram()
    return chi_square_distance(histogram1, histogram2)


class NeighborsQueue():
    """Helper class for maintaining a queue of neighbors.

    Note that although this uses a PriorityQueue internally, this class is not
    itself thread-safe! All get/put calls are non-blocking.

    To make this thread-safe, I believe we just need a lock in put() that
    ensures that the removal of the furthest neighbor and the insertion of the
    new neighbor is not intercepted by a different call to put / get."""
    def __init__(self, maxsize=None):
        self.pq = PriorityQueue(maxsize)

    def get(self):
        negative_distance, random_number, obj = self.pq.get_nowait()
        return obj, -negative_distance

    def empty(self):
        return self.pq.empty()

    def full(self):
        return self.pq.full()

    def qsize(self):
        return self.pq.qsize()

    def put(self, obj, distance):
        if self.pq.full():
            furthest_distance = -self.pq.queue[0][0]
            if distance < furthest_distance:
                self.pq.get_nowait()  # Remove the further neighbor
                # Add a random number so priority queue resolves distance ties
                # randomly rather than trying to compare objects.
                self.pq.put((-distance, random.random(), obj))
        else:
            self.pq.put((-distance, random.random(), obj))


