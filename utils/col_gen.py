import cplex as cp
import numpy as np

import cplex as cp
import numpy as np


def column_generation(L, l, b):
    """
    使用列生成算法求解切割库存问题
    L: 长卷长度
    l: 每种需求的长度列表
    b: 每种需求的需求量
    """
    # 初始化主问题 (RLPM)
    cplex_obj = cp.Cplex()
    cplex_obj.set_results_stream(None)  # 禁用输出流，保持控制台整洁
    cplex_obj.objective.set_sense(cplex_obj.objective.sense.minimize)

    # 初始化模式矩阵 A（每列是一个模式）
    num_items = len(l)
    A = np.eye(num_items)  # 初始模式：每列只切一类
    variable_names = [f"x{i + 1}" for i in range(num_items)]

    # 添加初始变量到主问题
    for i in range(num_items):
        cplex_obj.variables.add(names=[variable_names[i]], types=['C'], lb=[0])
        cplex_obj.linear_constraints.add(
            lin_expr=[[[variable_names[i]], [1.0]]],
            senses=["G"],
            rhs=[b[i]],
            names=[f"st{i + 1}"]
        )

    # 创建主问题
    cplex_obj.objective.set_linear([(variable_names[i], 1.0) for i in range(num_items)])

    iter = 0
    while True:
        iter += 1
        # 求解主问题 (RLPM)
        cplex_obj.solve()

        # 显式构建对偶问题
        cplex_obj_dual = cp.Cplex()
        cplex_obj_dual.set_results_stream(None)
        cplex_obj_dual.objective.set_sense(cplex_obj_dual.objective.sense.maximize)

        # 对偶问题目标函数：max b1*pi1 + b2*pi2 + ... + bm*pim
        dual_var_names = [f"pi{i + 1}" for i in range(num_items)]
        cplex_obj_dual.variables.add(
            names=dual_var_names,
            obj=b,
            types=['C'] * num_items,
            lb=[0] * num_items,
            ub=[1] * num_items  # pi <= 1 是对偶约束的一部分
        )

        # 对偶问题的约束： A^T * pi <= c
        for j in range(A.shape[1]):
            constraint = [[dual_var_names, A[:, j].tolist()]]
            cplex_obj_dual.linear_constraints.add(
                lin_expr=constraint,
                senses=['L'],
                rhs=[1.0]  # 对应主问题目标函数的系数为1
            )

        # 求解对偶问题
        cplex_obj_dual.solve()

        # 获取对偶解 pi
        pi = np.array(cplex_obj_dual.solution.get_values())

        # 初始化辅助子问题 (Auxiliary Problem)
        cplex_obj_aux = cp.Cplex()
        cplex_obj_aux.set_results_stream(None)
        aux_var_names = [f"z{i + 1}" for i in range(num_items)]
        for i in range(num_items):
            cplex_obj_aux.variables.add(names=[aux_var_names[i]], types=['C'], lb=[0], ub=[cp.infinity])

        # 设置辅助子问题目标函数
        cplex_obj_aux.objective.set_linear([(aux_var_names[i], pi[i]) for i in range(num_items)])
        cplex_obj_aux.objective.set_sense(cplex_obj_aux.objective.sense.maximize)

        # 添加约束：长度约束
        cplex_obj_aux.linear_constraints.add(
            lin_expr=[[aux_var_names, l]],
            senses=["L"], rhs=[L]
        )

        # 求解辅助子问题
        cplex_obj_aux.solve()

        # 获取新模式 z
        z = np.array(cplex_obj_aux.solution.get_values())
        # 对z进行向下取整
        z = np.floor(z)
        reduced_cost = 1 - np.dot(pi, z)

        x = cplex_obj.solution.get_values()
        obj_fct_val = np.sum(x)
        print_result(iter,
                     cplex_obj,
                     obj_fct_val,
                     x,
                     pi,
                     z,
                     reduced_cost)

        # 如果 reduced cost > 0，停止迭代
        if reduced_cost > -0.0001:
            break

        # 将新模式添加到模式矩阵 A
        A = np.column_stack((A, z))

        # 在主问题中添加新变量
        new_var_name = f"x{A.shape[1]}"
        cplex_obj.variables.add(names=[new_var_name], types=['C'], lb=[0], ub=[cp.infinity])
        for i in range(num_items):
            cplex_obj.linear_constraints.set_coefficients(i, new_var_name, z[i])

    # 获取最终解
    final_solution = cplex_obj.solution.get_values()

    # 对 x 向上取整
    final_solution = np.ceil(final_solution)

    # 计算最终目标值
    final_obj_value = np.sum(final_solution)

    return A, final_solution, final_obj_value




def print_result(iter, rlpm_cplex_obj ,rlpm_value, rlpm_solution, pi, z, k):

    print("==================================================")
    # 1st RLPM problem
    print(f"[N.{iter} RLPM problem] :")
    # 打印objective function
    var_names = rlpm_cplex_obj.variables.get_names()
    print(var_names)
    obj_fct = " + ".join(
        [f"{var_names[i]}" for i in range(len(var_names))]
    )
    print(f"min {obj_fct}")
    # 打印全部约束的表达式，用正常人可读的方式，比如x1 + x2 + x3 >= 30
    for i in range(rlpm_cplex_obj.linear_constraints.get_num()):
        row = rlpm_cplex_obj.linear_constraints.get_rows(i)
        indices, values = row.unpack()
        constraint_str = " + ".join(
            [f"{rlpm_cplex_obj.variables.get_names(index)}*{value}" for index, value in zip(indices, values)]
        )
        rhs = rlpm_cplex_obj.linear_constraints.get_rhs(i)
        print(f"st{i+1} : {constraint_str} >= {rhs}")

    print("-----------------------------------------------")
    # print the current value of the variables
    print("x = ", rlpm_solution)

    # print the current value of the objective function
    print(f"min({obj_fct}) = ", rlpm_value)

    print("-----------------------------------------------")

    # print the dual solution
    print("pi = ", pi)

    # print the solution of the 1st auxiliary sub-problem
    print("z = ", z)

    # print the reduced cost
    print("reduced cost = 1 - <pi, z> = ", k)




