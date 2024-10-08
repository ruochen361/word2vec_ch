# huffman_tree.py

from collections import Counter
import heapq

class Node:
    def __init__(self, weight, symbol=None, left=None, right=None):
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.weight < other.weight

def build_huffman_tree(frequencies):
    heap = [[weight, Node(weight, symbol)] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        merged = Node(lo[0] + hi[0], left=lo[1], right=hi[1])
        heapq.heappush(heap, [merged.weight, merged])
    return heap[0][1]

def get_paths(node, path='', paths=None):
    if paths is None:
        paths = {}
    if node.symbol is not None:
        paths[node.symbol] = path
    if node.left:
        get_paths(node.left, path + '0', paths)
    if node.right:
        get_paths(node.right, path + '1', paths)
    return paths

def build_huffman_paths(frequencies):
    tree = build_huffman_tree(frequencies)
    paths = get_paths(tree)
    return paths

def build_huffman_tree_and_paths(tokenized_text):
    freqs = Counter(tokenized_text)
    huffman_paths = build_huffman_paths(freqs)
    return huffman_paths