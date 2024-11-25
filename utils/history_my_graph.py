import math

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.vertices = list(graph.keys())
        self.edges = self.get_edges()

    def get_edges(self):
        edges = []
        visited = set()
        for i, vertex in enumerate(self.vertices):
            for j, weight in enumerate(self.graph[vertex]):
                if weight != 0:  # 0-weight edges means no connection
                    edge = tuple(
                        sorted(
                            (vertex, self.vertices[j])
                        )
                    )
                    if edge not in visited:
                        visited.add(edge)
                        edges.append(
                            (vertex, self.vertices[j], weight)
                        )
        return edges
# [('A', 'B', 34), ('A', 'C', 46), ('A', 'F', 19),
# ('B', 'E', 12),
# ('C', 'D', 17), ('C', 'F', 25),
# ('D', 'E', 38), ('D', 'F', 25),
# ('E', 'F', 26)]

    @staticmethod
    def graph_constructor(V, E):
        """Constructs a graph dictionary from a vertex list and edge list."""
        graph = {v: [0] * len(V) for v in V}
        for u, v, w in E:
            i, j = V.index(u), V.index(v)
            graph[u][j] = w
            graph[v][i] = w
        return graph

    def print_graph(self):
        for vertex, adj in self.graph.items():
            print(f"{vertex}: {adj}")

# find 函数功能：
# find 函数用于查找一个顶点所属的集合的代表元素（这个代表元素也被称为“根节点”）。
# 并查集中的每个集合都有一个唯一的代表元素。
# 实现细节：
#     如果一个顶点是自己集合的代表（即 parent[vertex] == vertex），直接返回这个顶点。
#     如果不是，递归地沿着父节点链向上查找，并将路径上的所有节点直接指向根节点（路径压缩优化）。
# find函数接受两个参数：parent 和 vertex。
# parent 是一个字典，用于存储每个顶点的父节点。
# vertex 是要查找的顶点。

    def find_root(self, parent, vertex):
        if parent[vertex] == vertex:
            return vertex
        return self.find_root(parent, parent[vertex])

# union 函数功能：
# union 函数用于将两个不同集合合并成一个集合。主要用于在添加边时更新并查集状态。
# 实现细节：
#   - 找到两个顶点各自的根节点（代表元素）。
#   - 根据集合的秩（rank）决定合并策略：
#       - 将秩较低的集合的根节点挂到秩较高的集合的根节点下。
#       - 如果两个集合的秩相同，则任选一个作为新根，并将其秩加 1。
#
# 优化：按秩合并（Union by Rank）
#     按秩合并通过始终让树的高度尽量小，减少后续 find 操作的深度，从而提高效率。

    def union(self, parent, rank, v1, v2):
        root1 = self.find_root(parent, v1)
        root2 = self.find_root(parent, v2)

        if rank[root1] < rank[root2]: # rank[root1] < rank[root2]，将 root1 挂到 root2 下, 且不改变两个根节点的秩
            parent[root1] = root2
        elif rank[root1] > rank[root2]: # rank[root1] > rank[root2]，将 root2 挂到 root1 下, 且不改变两个根节点的秩
            parent[root2] = root1
        else: # rank[root1] == rank[root2]，将 root1 挂到 root2 下, 且 root2 的秩加 1
            parent[root2] = root1
            rank[root1] += 1

    def kruskal(self):
        mst = []
        parent = {}
        rank = {}

        # initialise parent and rank
        for vertex in self.vertices:
            parent[vertex] = vertex
            rank[vertex] = 0
        # Init. parent:  {'A': 'A',
        #                 'B': 'B',
        #                 'C': 'C',
        #                 'D': 'D',
        #                 'E': 'E',
        #                 'F': 'F'}
        # Init. rank:  {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}

        # sort edges by weight
        edges = sorted(self.edges, key=lambda x: x[2])

        for e in edges:
            v1, v2, w = e

            # check if including this edge in the mst will form a cycle
            if self.find_root(parent, v1) != self.find_root(parent, v2):
                mst.append(e)
                self.union(parent, rank, v1, v2)
                # print("Parent: ", parent)
                # print("Rank: ", rank)
            # Parent: {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'B', 'F': 'F'}
            # Rank: {'A': 0, 'B': 1, 'C': 0, 'D': 0, 'E': 0, 'F': 0}

            # Parent: {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'C', 'E': 'B', 'F': 'F'}
            # Rank: {'A': 0, 'B': 1, 'C': 1, 'D': 0, 'E': 0, 'F': 0}

            # Parent: {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'C', 'E': 'B', 'F': 'A'}
            # Rank: {'A': 1, 'B': 1, 'C': 1, 'D': 0, 'E': 0, 'F': 0}

            # Parent: {'A': 'C', 'B': 'B', 'C': 'C', 'D': 'C', 'E': 'B', 'F': 'A'}
            # Rank: {'A': 1, 'B': 1, 'C': 2, 'D': 0, 'E': 0, 'F': 0}

            # Parent: {'A': 'C', 'B': 'C', 'C': 'C', 'D': 'C', 'E': 'B', 'F': 'A'}
            # Rank: {'A': 1, 'B': 1, 'C': 2, 'D': 0, 'E': 0, 'F': 0}

        return mst


# e=8
# v=6
# print(e * math.log2(e))
# print(v ** 2)
#
#
# graph = {
#     'A': [0, 0.34, 0.46, 0, 0, 0.19],
#     'B': [0.34, 0, 0, 0, 0.12, 0 ],
#     'C': [0.46, 0, 0, 0.17, 0, 0.25],
#     'D': [0, 0, 0.17, 0, 0.38, 0.25],
#     'E': [0, 0.12, 0, 0.25, 0, 0.26],
#     'F': [0.19, 0, 0.25, 0.25, 0.26,0],
# }
#
# V = ['A', 'B', 'C', 'D', 'E', 'F']
# E = [('A', 'B', 0.34), ('A', 'C', 0.46), ('A', 'F', 0.19),
#      ('B', 'E', 0.12),
#      ('C', 'D', 0.17), ('C', 'F', 0.25),
#      ('D', 'E', 0.38), ('D', 'F', 0.25),
#      ('E', 'F', 0.26)]
#
#
#
# print(graph['A'])
# for j, v in enumerate(graph['A']):
#     print(j, v)
#
# g = Graph(graph)
#
# mst = g.kruskal()
#
# print("Edges in the Minimum Spanning Tree:")
# for edge in mst:
#     print(f"{edge[0]} -- {edge[1]} == {edge[2]}")
#
# # Edges in the Minimum Spanning Tree:
# # B -- E == 0.12
# # C -- D == 0.17
# # A -- F == 0.19
# # C -- F == 0.25
# # E -- F == 0.26
#
# # 如果碰到A-B和B-A，那么需要把这两个边认定为同一个边
#
# g2 = Graph.graph_constructor(V, E)
# print("g2: ", g2)
#
# gg2 = Graph(g2)
#
# # 输出图的邻接矩阵形式
# print("\nAdjacency matrix representation:")
# gg2.print_graph()
#
# print("===============================================================")
# V_aux = [u for u in V]
# E_aux = [e for e in E]
# graph_aux = Graph.graph_constructor(V_aux, E_aux)
# g_aux_obj = Graph(graph_aux)
# p_new = g_aux_obj.kruskal()
# p_new_binary = [1 if e in p_new else 0 for e in E]
# print("E: ")
# for e in E:
#     print(e)
#
# print("p_new: ")
# for e in p_new:
#     print(e)
#
# print("p_new_binary: ", p_new_binary)
#
# print("===============================================================")
# p_new_pairs = [(u, v) for u, v, _ in p_new]
# E_pairs = [(u, v) for u, v, _ in E]
#
# print("p_new_pairs: ")
# for pair in p_new_pairs:
#     print(pair)
# print("E_pairs: ")
# for pair in E_pairs:
#     print(pair)
#
# p_new_binary = [1 if pair in p_new_pairs else 0 for pair in E_pairs]
# print("p_new_binary: ", p_new_binary)