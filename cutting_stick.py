from utils.col_gen_cut_stirck import column_generation

L = 259
l = [81, 70, 68]
b = [44, 3, 48]

A, final_solution, final_obj_value = column_generation(L, l, b)

print("===============================================================")
print("最终模式矩阵 A:")
print(A)
print("最终解 x:", final_solution)
print("最终目标值:", final_obj_value)
