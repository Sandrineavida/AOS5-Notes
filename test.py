import cplex as cp 
# 创建cplex对象 
cplex_obj = cp.Cplex() 
# 创建连续值变量 
x = cplex_obj.variables.add(names=['x'], types=['C'], lb=[0], ub=[10]) 
y = cplex_obj.variables.add(names=['y'], types=['C'], lb=[0], ub=[10]) 
# 创建目标函数（线性） 
cplex_obj.objective.set_linear([('x', 1.0), ('y', 2.0)]) 
# 设置优化方向（最大化） 
cplex_obj.objective.set_sense(cplex_obj.objective.sense.maximize) 
# 添加约束 
cplex_obj.linear_constraints.add(lin_expr=[[['x', 'y'], [1.0, 2.0]]], senses=['L'], rhs=[30], names=['st1']) 
cplex_obj.linear_constraints.add(lin_expr=[[['x', 'y'], [1.0, 1.0]]], senses=['L'], rhs=[15], names=['st2'])

# rhs全称是right-hand side，表示约束的右侧值
# st1 : x + 2y <= 30
print("==================================================")

# 设置名称，测试set_names和get_names
print(cplex_obj.linear_constraints.get_names(1))
cplex_obj.linear_constraints.set_names(1,'st3')
print(cplex_obj.linear_constraints.get_names(1))

print("==================================================")

# 打印objective function
var_names = cplex_obj.variables.get_names()
print(var_names)
obj_fct = " + ".join(
    [f"{var_names[i]}*{cplex_obj.objective.get_linear(i)}" for i in range(len(var_names))]
)
print(f"max : {obj_fct}")

# 打印全部约束的表达式，用人类可读的方式，比如x + 2y <= 30
row = cplex_obj.linear_constraints.get_rows(0)
print(row)
print(row.ind)
indices, values = row.unpack()
constraint_str = " + ".join(
    [f"{value}*{cplex_obj.variables.get_names(index)}" for index, value in zip(indices, values)]
)
rhs = cplex_obj.linear_constraints.get_rhs(0)
print(f"st1 : {constraint_str} <= {rhs}")

print("==================================================")
# 求解问题 
cplex_obj.solve() 
# 检查求解可行性 
if cplex_obj.solution.is_primal_feasible(): 
    print("Solution is feasible") 
    # 获取解 
    solution = cplex_obj.solution.get_values() 
    print(solution) 
else: 
    print("Solution is infeasible")



# import cplex as cp
# import numpy as np
#
# cplex_obj = cp.Cplex()
#
# # L = 218 cm
# # l1 = 81 cm, b1 = 44 (number of pieces)
# # l2 = 70 cm, b2 = 3
# # l3 = 68 cm, b3 = 48
#
# # 1st RLPM:
# # a1: pattern 1 = (1,0,0).T (meaning: cut 1 long roll into 1 short roll of length l1)
# # a2: pattern 2 = (0,1,0).T (meaning: cut 1 long roll into 1 short roll of length l2)
# # a3: pattern 3 = (0,0,1).T (meaning: cut 1 long roll into 1 short roll of length l3)
# #              min x1 + x2 + x3
# #             s.t. x1 >= b1 = 44
# #                  x2 >= b2 = 3
# #                  x3 >= b3 = 48
# #                   x >= 0, integer
#
# # create variables (I->C)
# # 将主问题（RLPM）设置为线性规划（LP），通过求解松弛问题来获得对偶解
# x1 = cplex_obj.variables.add(names=['x1'], types=['C'], lb=[0])
# x2 = cplex_obj.variables.add(names=['x2'], types=['C'], lb=[0])
# x3 = cplex_obj.variables.add(names=['x3'], types=['C'], lb=[0])
#
# # create objective function
# cplex_obj.objective.set_linear([('x1', 1.0), ('x2', 1.0), ('x3', 1.0)])
# # set optimization direction for the objective function
# cplex_obj.objective.set_sense(cplex_obj.objective.sense.minimize)
#
# # add constraints
# cplex_obj.linear_constraints.add(lin_expr=[[['x1'], [1.0]]], senses=['G'], rhs=[44], names=['st1'])
# cplex_obj.linear_constraints.add(lin_expr=[[['x2'], [1.0]]], senses=['G'], rhs=[3], names=['st2'])
# cplex_obj.linear_constraints.add(lin_expr=[[['x3'], [1.0]]], senses=['G'], rhs=[48], names=['st3'])
#
# # solve the dual problem to get the dual solution pi = (pi1, pi2, pi3).T
# # ps: in cplex, 求解对偶问题和对偶最优解的过程通常是通过 直接求解原问题（主问题）来实现的
# cplex_obj.solve()
# # pi = cplex_obj.solution.get_dual_values()
#
# # Define the dual problem
# #     max pi1*b1 + pi2*b2 + pi3*b3
# #     s.t. pi1 <= 1
# #          pi2 <= 1
# #          pi3 <= 1
# #          pi1, pi2, pi3 >= 0
#
# cplex_obj_dual = cp.Cplex()
#
# # create variables
# pi1 = cplex_obj_dual.variables.add(names=['pi1'], types=['C'], lb=[0])
# pi2 = cplex_obj_dual.variables.add(names=['pi2'], types=['C'], lb=[0])
# pi3 = cplex_obj_dual.variables.add(names=['pi3'], types=['C'], lb=[0])
#
# # create objective function
# cplex_obj_dual.objective.set_linear([('pi1', 44), ('pi2', 3), ('pi3', 48)])
# # set optimization direction for the objective function
# cplex_obj_dual.objective.set_sense(cplex_obj_dual.objective.sense.maximize)
#
# # add constraints
# cplex_obj_dual.linear_constraints.add(lin_expr=[[['pi1'], [1.0]]], senses=['L'], rhs=[1], names=['st1'])
# cplex_obj_dual.linear_constraints.add(lin_expr=[[['pi2'], [1.0]]], senses=['L'], rhs=[1], names=['st2'])
# cplex_obj_dual.linear_constraints.add(lin_expr=[[['pi3'], [1.0]]], senses=['L'], rhs=[1], names=['st3'])
#
# # solve the dual problem
# cplex_obj_dual.solve()
#
# # get the dual solution
# pi = cplex_obj_dual.solution.get_values()
#
# # get the current value of the objective function
# obj_value = cplex_obj.solution.get_objective_value()
#
# # get the current value of the variables
# solution = cplex_obj.solution.get_values()
#
# # 检查主问题求解状态
# print("RLPM solve status:", cplex_obj.solution.get_status_string())
#
# # 1st auxiliary sub-problem:
# #      max pi1*z1 + pi2*z2 + pi3*z3
# #      s.t. l1*z1 + l2*z2 + l3*z3 <= L
# #           z1, z2, z3 >= 0, integer
#
# cplex_obj_aux = cp.Cplex()
#
# # create variables
# z1 = cplex_obj_aux.variables.add(names=['z1'], types=['C'], lb=[0])
# z2 = cplex_obj_aux.variables.add(names=['z2'], types=['C'], lb=[0])
# z3 = cplex_obj_aux.variables.add(names=['z3'], types=['C'], lb=[0])
#
# # create objective function
# cplex_obj_aux.objective.set_linear([('z1', pi[0]), ('z2', pi[1]), ('z3', pi[2])])
# # set optimization direction for the objective function
# cplex_obj_aux.objective.set_sense(cplex_obj_aux.objective.sense.maximize)
#
# # add constraints
# cplex_obj_aux.linear_constraints.add(
#     lin_expr=[
#     [['z1', 'z2', 'z3'], [81, 70, 68]]
#     ],
#     senses=['L'], rhs=[218], names=['st1']
# )
#
# # solve the 1st auxiliary sub-problem
# cplex_obj_aux.solve()
#
# # get the solution of the 1st auxiliary sub-problem
# z = cplex_obj_aux.solution.get_values()
#
# # 检查子问题求解状态
# print("Auxiliary problem solve status:", cplex_obj_aux.solution.get_status_string())
#
# # calculate the reduced cost
# pi = np.array(pi)
# z = np.array(z)
# k = 1 - np.dot(pi, z)
# k_ceil = 1 - np.dot(pi, np.ceil(z))
#
# print("==================================================")
# # 1st RLPM problem
# print("[1st RLPM problem] :")
# print("min x1 + x2 + x3")
# print("s.t. x1 >= 44")
# print("     x2 >= 3")
# print("     x3 >= 48")
# print("     x >= 0, integer")
# print("-----------------------------------------------")
# # print the current value of the objective function
# print("min(x1+x2+x3) = ", obj_value)
#
# # print the current value of the variables
# print("[x1, x2, x3] = ", solution)
#
# print("-----------------------------------------------")
#
# # print the dual solution
# print("pi = ", pi)
#
# # print the solution of the 1st auxiliary sub-problem
# print("z = ", z)
#
# # print 向下取整的z
# print("floor(z) = ", np.floor(z))
#
# # print the reduced cost
# print("reduced cost = ", k)
#
# # print the reduced cost (floor)
# print("reduced cost (ceil) = ", k_ceil)
#
# print("==================================================")
#
# from utils.col_gen import print_result
#
# print_result(1, cplex_obj, obj_value, solution, pi, z, k, k_ceil)
#
#
#
#
# print("==================================================")
#
#