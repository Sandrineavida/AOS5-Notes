import math
import heapq

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.vertices = list(graph.keys())
        self.edges = self.get_edges()

    def get_edges(self):
        edges = []
        # visited = set()
        for i, vertex in enumerate(self.vertices):
            for j, weight in enumerate(self.graph[vertex]):
                if weight != 0:  # 0-weight edges means no connection
                    edges.append(
                        (vertex, self.vertices[j], weight)
                    )
                    # edge = tuple(
                    #     sorted(
                    #         (vertex, self.vertices[j])
                    #     )
                    # )
                    # if edge not in visited:
                    #     visited.add(edge)
                    #     edges.append(
                    #         (vertex, self.vertices[j], weight)
                    #     )
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

    def dijkstra(self, start):
        """Finds shortest paths from the start vertex to all other vertices."""
        distances = {vertex: float('inf') for vertex in self.vertices}
        previous = {vertex: None for vertex in self.vertices}
        distances[start] = 0

        priority_queue = [(0, start)]  # (distance, vertex)

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            # Skip processing if the distance is already greater than the known shortest
            if current_distance > distances[current_vertex]:
                continue

            for neighbor_index, weight in enumerate(self.graph[current_vertex]):
                if weight != 0:  # There's a connection
                    neighbor = self.vertices[neighbor_index]
                    distance = current_distance + weight

                    # If found a shorter path to the neighbor
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(priority_queue, (distance, neighbor))

        return distances, previous

    def dijkstra_with_end_vertex(self, start, end):
        """Finds the shortest path from start to end using Dijkstra's algorithm."""
        distances, previous = self.dijkstra(start)

        # Reconstruct the path from start to end
        path = []
        path_paired = []
        current_vertex = end
        while current_vertex is not None:
            path.insert(0, current_vertex)
            if previous[current_vertex] is not None:
                path_paired.insert(0, (previous[current_vertex], current_vertex))
            current_vertex = previous[current_vertex]

        # If the start vertex is not reachable to the end, return empty path
        if distances[end] == float('inf'):
            return [], float('inf'), []

        return path, distances[end], path_paired


# Example usage
if __name__ == "__main__":
    V = ['A', 'B', 'C', 'D', 'E']
    E = [
        ('A', 'B', 10),
        ('A', 'D', 30),
        ('A', 'E', 100),
        ('B', 'C', 50),
        ('C', 'E', 10),
        ('D', 'C', 20),
        ('D', 'E', 60)
    ]

    graph_dict = Graph.graph_constructor(V, E)
    graph = Graph(graph_dict)

    graph.print_graph()

    # Dijkstra from a single start vertex to all others
    print("\nDijkstra from A:")
    distances, previous = graph.dijkstra('A')
    print("Distances:", distances)
    print("Previous vertices:", previous)

    # Dijkstra from A to E
    print("\nShortest path from A to E:")
    path, distance, path_paired = graph.dijkstra_with_end_vertex('A', 'E')
    print("Path:", path)
    print("Distance:", distance)
    print("Paired path:")
    for pair in path_paired:
        print(pair)

    E_pairs = [(u, v) for u, v, _ in E]
    p_new_binary = [1 if pair in path_paired else 0 for pair in E_pairs]
    print("p_new_binary: ", p_new_binary)

# if __name__ == "__main__":
#     e=8
#     v=6
#     print(e * math.log2(e))
#     print(v ** 2)
#
#
#     graph = {
#         'A': [0, 0.34, 0.46, 0, 0, 0.19],
#         'B': [0.34, 0, 0, 0, 0.12, 0 ],
#         'C': [0.46, 0, 0, 0.17, 0, 0.25],
#         'D': [0, 0, 0.17, 0, 0.38, 0.25],
#         'E': [0, 0.12, 0, 0.25, 0, 0.26],
#         'F': [0.19, 0, 0.25, 0.25, 0.26,0],
#     }
#
#     V = ['A', 'B', 'C', 'D', 'E', 'F']
#     E = [('A', 'B', 0.34), ('A', 'C', 0.46), ('A', 'F', 0.19),
#          ('B', 'E', 0.12),
#          ('C', 'D', 0.17), ('C', 'F', 0.25),
#          ('D', 'E', 0.38), ('D', 'F', 0.25),
#          ('E', 'F', 0.26)]
#
#
#
#     print(graph['A'])
#     for j, v in enumerate(graph['A']):
#         print(j, v)
#
#     g = Graph(graph)
#
#     mst = g.kruskal()
#
#     print("Edges in the Minimum Spanning Tree:")
#     for edge in mst:
#         print(f"{edge[0]} -- {edge[1]} == {edge[2]}")
#
#     # Edges in the Minimum Spanning Tree:
#     # B -- E == 0.12
#     # C -- D == 0.17
#     # A -- F == 0.19
#     # C -- F == 0.25
#     # E -- F == 0.26
#
#     # 如果碰到A-B和B-A，那么需要把这两个边认定为同一个边
#
#     g2 = Graph.graph_constructor(V, E)
#     print("g2: ", g2)
#
#     gg2 = Graph(g2)
#
#     # 输出图的邻接矩阵形式
#     print("\nAdjacency matrix representation:")
#     gg2.print_graph()
#
#     print("===============================================================")
#     V_aux = [u for u in V]
#     E_aux = [e for e in E]
#     graph_aux = Graph.graph_constructor(V_aux, E_aux)
#     g_aux_obj = Graph(graph_aux)
#     p_new = g_aux_obj.kruskal()
#     p_new_binary = [1 if e in p_new else 0 for e in E]
#     print("E: ")
#     for e in E:
#         print(e)
#
#     print("p_new: ")
#     for e in p_new:
#         print(e)
#
#     print("p_new_binary: ", p_new_binary)
#
#     print("===============================================================")
#     p_new_pairs = [(u, v) for u, v, _ in p_new]
#     E_pairs = [(u, v) for u, v, _ in E]
#
#     print("p_new_pairs: ")
#     for pair in p_new_pairs:
#         print(pair)
#     print("E_pairs: ")
#     for pair in E_pairs:
#         print(pair)
#
#     p_new_binary = [1 if pair in p_new_pairs else 0 for pair in E_pairs]
#     print("p_new_binary: ", p_new_binary)