#!/usr/bin/env python


# Segment Tree data structure; see: https://en.wikipedia.org/wiki/Segment_tree
class SegmentTree():
    def __init__(self, size):
        # index of next data to be added
        self.index = 0
        # max number of items
        self.size = size
        # indicates if segment tree is full
        self.full = False
        # initial tree with all priorities set to 0
        self.sum_tree = [0] * (2 * size - 1)
        # data storage
        self.data = [None] * size

        # initial maximal priority
        self.max = 1

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Search for a value in sum tree
    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    # Returns total sum in tree == sum in root node
    def total(self):
        return self.sum_tree[0]

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)

    # Adds a leaf to the tree
    def append(self, data, value):
        self.data[self.index] = data
        self.update(self.index + self.size - 1, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)