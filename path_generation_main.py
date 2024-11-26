from utils.col_gen_path_generation import column_generation
import numpy as np

V = ['A', 'B', 'C', 'D', 'E']
# 边集合
E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E'), ('D', 'E')]


# 初始路径集合的 0-1 矩阵
# Pd = [
#     [1, 1, 0, 0, 0, 1, 0],  # A -> B
#     [0, 0, 1, 0, 1, 0, 1],  # A -> C
#     [1, 0, 0, 0, 0, 0, 1],  # B -> C
#     [0, 1, 0, 0, 0, 0, 0],  # B -> D
#     [0, 0, 1, 0, 0, 0, 0],  # C -> D
#     [1, 1, 1, 1, 1, 0, 0],  # C -> E
#     [0, 1, 1, 1, 1, 1, 1],  # D -> E
# ]

Pd = [
    [1],  # A -> B
    [0],  # A -> C
    [1],  # B -> C
    [0],  # B -> D
    [0],  # C -> D
    [1],  # C -> E
    [0],  # D -> E
]

d = ['A', 'E']

Cap_E = [10,
         15,
         5,
         20,
         10,
         10,
         25
]
# 节点集合
# V = ['A', 'B', 'C', 'D', 'E', 'F']
#
# # 边集合
# E = [
#     ('A', 'B'), ('A', 'C'), ('A', 'F'),
#     ('B', 'C'), ('B', 'D'), ('B', 'F'),
#     ('C', 'D'), ('C', 'E'), ('C', 'F'),
#     ('D', 'E'), ('D', 'F'),
#     ('E', 'F')
# ]
#
# # 初始路径集合的 0-1 矩阵
# Pd = [
#     [1],  # A -> B  1
#     [0],  # A -> C  2
#     [0],  # A -> F  3
#     [1],  # B -> C  4
#     [0],  # B -> D  5
#     [0],  # B -> F  6
#     [0],  # C -> D  7
#     [0],  # C -> E  8
#     [0],  # C -> F  9
#     [0],  # D -> E  10
#     [0],  # D -> F  11
#     [0],  # E -> F  12
# ]
#
# # 边的容量
# Cap_E = [
#     10,  # A -> B
#     15,  # A -> C
#     20,  # A -> F
#     5,   # B -> C
#     20,  # B -> D
#     15,  # B -> F
#     10,  # C -> D
#     10,  # C -> E
#     15,  # C -> F
#     25,  # D -> E
#     15,  # D -> F
#     10   # E -> F
# ]
#
# # 需求
# d = ['A', 'E']


Pd0 = np.array(Pd)

# for i, row in enumerate(Pd0):
#     print(Cap_E[i])


Pd, final_solution, final_obj_value = column_generation(d, V, E, Cap_E, Pd0)

print("===============================================================")
print("Final Pattern Matrix:")
print(Pd)

print("Final Solution:")
print(final_solution)

print("Final Objective Value:")
print(final_obj_value)