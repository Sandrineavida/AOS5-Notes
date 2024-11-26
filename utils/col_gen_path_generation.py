import cplex as cp
import numpy as np
from utils import my_graph
from utils.col_gen_visual import visualize_process_with_legend

def column_generation(d, V, E, Cap_E, Pd0):
    """
    使用列生成算法求解切Path Generation问题
    d: demand, (sd, td) ; e.g. (A, E) (A和E之间必须联通)
    V : vertices
    E : edges
    Cap_E : Maximum capacity of each edge
    """

    sd = d[0]
    td = d[1]

    # 初始化主问题 (RLPM)
    cplex_obj = cp.Cplex()
    cplex_obj.set_results_stream(None)  # 禁用输出流，保持控制台整洁
    cplex_obj.objective.set_sense(cplex_obj.objective.sense.maximize)

    # 初始化模式矩阵 Pd（每列是一种路径）
    num_items = Pd0.shape[1]
    num_dual_items = len(E)
    Pd = Pd0

    variable_names = [f"xp{i + 1}" for i in range(num_items)]

    # 添加初始变量到主问题
    for i in range(num_items):
        cplex_obj.variables.add(names=[variable_names[i]], types=['C'], lb=[0])

    # 以Pd为系数，添加约束
    for i, row in enumerate(Pd):
        cplex_obj.linear_constraints.add(
            lin_expr=[cp.SparsePair(ind=variable_names, val=row.tolist())],
            senses=["L"],  # "L" 表示小于等于 (<=)
            rhs=[Cap_E[i]],
            names = [f"st{i + 1}"]
        )

    # 创建主问题
    cplex_obj.objective.set_linear([(variable_names[i], 1.0) for i in range(num_items)])

    iter = 0
    while True:
        iter += 1
        # 求解主问题 (RLPM)
        cplex_obj.solve()
        xp = cplex_obj.solution.get_values()
        obj_fct_val = np.sum(xp)
        print("==================iter [", iter, "]======================")
        print("xp = ", xp)
        print("obj_fct_val = ", obj_fct_val)

        # 显式构建对偶问题
        cplex_obj_dual = cp.Cplex()
        cplex_obj_dual.set_results_stream(None)
        cplex_obj_dual.objective.set_sense(cplex_obj_dual.objective.sense.minimize)

        # 对偶问题目标函数：min c1*pi1 + c2*pi2 + ... + ce*pie
        dual_var_names = [f"pi{i + 1}" for i in range(num_dual_items)]
        cplex_obj_dual.variables.add(
            names=dual_var_names,
            obj=Cap_E, # 定义变量的目标函数系数
            types=['C'] * num_dual_items,
            lb=[1e-6] * num_dual_items,
            # ub=[1] * num_dual_items  # pi <= 1, 把pi缩放到[0, 1]范围内，因为我们希望pi可以表示一个概率、一种权重
        )

        # 对偶问题的约束： A^T * pi >= 1， A=Pd
        for j in range(Pd.shape[1]):
            constraint = [[dual_var_names, Pd[:, j].tolist()]]
            cplex_obj_dual.linear_constraints.add(
                lin_expr=constraint,
                senses=['G'],
                rhs=[1.0]  # 对应主问题目标函数的系数为1
            )

        # 求解对偶问题
        cplex_obj_dual.solve()

        # 获取对偶解 pi
        pi = np.array(cplex_obj_dual.solution.get_values())

        # 初始化辅助子问题 (Auxiliary Problem)
        V_aux = [u for u in V]
        # E_aux = [(u, v, pi[i]) for i, (u, v) in enumerate(E)]
        E_aux = [(u, v, max(pi[i], 1e-6)) for i, (u, v) in enumerate(E)]
        graph_aux = my_graph.Graph.graph_constructor(V_aux, E_aux)
        g_aux_obj = my_graph.Graph(graph_aux)
        # _, _, p_new = g_aux_obj.dijkstra_with_end_vertex(sd, td)
        #
        # p_new_binary = [1 if pair in p_new else 0 for pair in E]

        path, distance, path_paired = g_aux_obj.dijkstra_with_end_vertex(sd, td)  # 从辅助问题求解路径
        # print("Path paired from auxiliary problem:", path_paired)

        # 生成二进制表示
        p_new_binary = [1 if pair in path_paired else 0 for pair in E]
        # print("p_new_binary:", p_new_binary)

        reduced_cost = 1 - np.dot(pi, p_new_binary)


        print_result(iter,
                     cplex_obj,
                     obj_fct_val,
                     xp,
                     pi,
                     p_new_binary,
                     reduced_cost)

        visualize_process_with_legend(V, E, Cap_E, Pd, xp)

        # 如果 reduced cost < 0，停止迭代
        if reduced_cost < 0.0001:
            break

        # 将新模式添加到模式矩阵 A=Pd
        Pd = np.column_stack((Pd, p_new_binary))
        print(Pd)

        # 在主问题中添加新变量
        new_var_name = f"xp{Pd.shape[1]}"
        cplex_obj.variables.add(names=[new_var_name], types=['C'], lb=[0], ub=[cp.infinity])

        # print("!!!!!!!!!!!!! ", cplex_obj.objective.get_linear())
        # 更新目标函数
        cplex_obj.objective.set_linear([(new_var_name, 1.0)])
        # print("?????????????????????? ", cplex_obj.objective.get_linear())

        # 修改约束，将新变量添加到所有边的容量约束中
        for i in range(len(E)):  # 遍历所有边（约束行）
            coefficient = p_new_binary[i]  # 新路径对应当前边的二进制值
            if coefficient != 0:  # 只有当系数非零时才需要更新约束
                cplex_obj.linear_constraints.set_coefficients(i, new_var_name, coefficient)

    # 获取最终解
    final_solution = cplex_obj.solution.get_values()

    # 对 x 向下取整
    final_solution = np.floor(final_solution)

    # 计算最终目标值
    final_obj_value = np.sum(final_solution)

    return Pd, final_solution, final_obj_value




def print_result(iter, rlpm_cplex_obj ,rlpm_value, rlpm_solution, pi, p_new_binary, k):

    print("==================================================")
    # 1st RLPM problem
    print(f"[N.{iter} RLPM problem] :")
    # 打印objective function
    var_names = rlpm_cplex_obj.variables.get_names()
    print(var_names)
    obj_fct = " + ".join(
        [f"{var_names[i]}" for i in range(len(var_names))]
    )
    print(f"max {obj_fct}")
    # 打印全部约束的表达式，用正常人可读的方式，比如x1 + x2 + x3 >= 30
    for i in range(rlpm_cplex_obj.linear_constraints.get_num()):
        row = rlpm_cplex_obj.linear_constraints.get_rows(i)
        indices, values = row.unpack()
        constraint_str = " + ".join(
            [f"{rlpm_cplex_obj.variables.get_names(index)}*{value}" for index, value in zip(indices, values)]
        )
        rhs = rlpm_cplex_obj.linear_constraints.get_rhs(i)
        print(f"st{i+1} : {constraint_str} <= {rhs}")

    print("-----------------------------------------------")
    # print the current value of the variables
    print("xp = ", rlpm_solution)

    # print the current value of the objective function
    print(f"max({obj_fct}) = ", rlpm_value)

    print("-----------------------------------------------")

    # print the dual solution
    print("pi = ", pi)

    # print the solution of the 1st auxiliary sub-problem
    print("p_new_binary = ", p_new_binary)

    # print the reduced cost
    print("reduced cost = 1 - <pi, p_new> = ", k)




