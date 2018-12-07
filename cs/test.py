"""Tests for the Clustering functions."""


import unittest
from alban import *


class TestSimpleClustering(unittest.TestCase):

    def test_empty(self):
        data = [[], []]
        self.assertEqual(clustering(data[0], simple_cluster), data[1])

    def test_one0(self):
        data = [[[0]], []]
        self.assertEqual(clustering(data[0], simple_cluster), data[1])

    def test_one1(self):
        data = [[[1]], [[[0, 0]]]]
        self.assertEqual(clustering(data[0], simple_cluster), data[1])

    def test_0s(self):
        data = [[[0,0,0,0],
                 [0,0,0,0],
                 [0,0,0,0],
                 [0,0,0,0]
                ],
                []
               ]
        self.assertEqual(clustering(data[0], simple_cluster), data[1])

    def test_1s(self):
        data = [[[1,1,1,1],
                 [1,1,1,1],
                 [1,1,1,1],
                 [1,1,1,1]
                ],
                [[[0,0],[1,0],[2,0],[3,0],
                  [3,1],[2,1],[1,1],[0,1],
                  [0,2],[1,2],[2,2],[3,2],
                  [3,3],[2,3],[1,3],[0,3]
                 ]
                ]
               ]
        self.assertEqual(clustering(data[0], simple_cluster), data[1])

    def test_example1(self):
        data = [[[0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,1,1,1,1,1,0,0,0,0,0,0,0],
                 [0,0,1,1,0,0,0,0,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,1,1,0,0,0,0,1,0],
                 [0,0,0,0,1,1,1,0,0,0,1,1,0],
                 [0,0,0,0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0]
                ],
                [[[1,1],[2,1],[3,1],[4,1],[5,1],[3,2],[2,2],[3,3]],
                 [[11,8],[11,9],[11,10],[10,10],[11,11]],
                 [[4,9],[5,9],[6,9],[6,10],[5,10],[4,10]]
                ]
               ]
        self.assertEqual(clustering(data[0], simple_cluster), data[1])


if __name__ == "__main__":
    unittest.main()
