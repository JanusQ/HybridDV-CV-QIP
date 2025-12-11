import os
import numpy as np
import json
from itertools import product
from datetime import datetime
import argparse

def generate_random_ip_instance(num_variables, num_constraints, max_coeff=5, max_rhs=30, N=10, min_feasible_count=10, max_feasible_count=70):
    """
    随机生成整数规划问题实例：A x = b
    修改版，增加了最大可行解数量限制
    
    Args:
        num_variables: 变量数量 (d)
        num_constraints: 约束数量 (m)
        max_coeff: 系数矩阵A的最大绝对值
        max_rhs: 右侧向量b的最大绝对值
        N: 可行解的最大值限制
        min_feasible_count: 要求的最小可行解数量
        max_feasible_count: 要求的最大可行解数量
    
    Returns:
        A, b, c, num_feasible: 约束矩阵、右侧向量、目标函数系数、可行解数量
    """
    
    attempts = 0
    max_attempts = 1000  # 最大尝试次数，防止无限循环
    
    while attempts < max_attempts:
        attempts += 1
        
        # 生成随机系数矩阵A (整数)
        A = np.random.randint(1, max_coeff + 1, size=(num_constraints, num_variables))
        
        # 生成随机可行解x0，确保x0中的每个元素都小于N
        x0 = np.random.randint(1, min(N, 5), size=num_variables)
        
        # 计算b = A x0，确保存在可行解
        b = A @ x0
        
        # 生成随机目标函数系数c
        c = np.random.randint(1, max_coeff + 1, size=num_variables)
        
        # 计算可行解数量
        num_feasible = 0
        feasible_solutions = []
        
        # 只在变量数量较少时计算精确的可行解数量
        if num_variables <= 6:  # 6个变量在N=7时大约有117,649种组合
            for x_candidate in product(range(N), repeat=num_variables):
                if np.allclose(A @ np.array(x_candidate), b):
                    num_feasible += 1
                    # 如果超过最大可行解数量，提前终止计算
                    if num_feasible > max_feasible_count:
                        break
            
            # 检查可行解数量是否在要求范围内
            if min_feasible_count <= num_feasible <= max_feasible_count:
                print(f"成功生成问题实例: {num_variables}变量, {num_constraints}约束, 可行解数量={num_feasible}, 尝试次数={attempts}")
                return A.tolist(), b.tolist(), c.tolist(), num_feasible
        else:
            # 对于变量数量较多的情况，至少检查是否有最小可行解数量
            # 但由于计算量大，我们无法精确计算是否超过最大可行解数量
            for x_candidate in product(range(N), repeat=num_variables):
                if np.allclose(A @ np.array(x_candidate), b):
                    feasible_solutions.append(x_candidate)
                    if len(feasible_solutions) >= min_feasible_count:
                        # 对于多变量情况，我们只能确保有足够的可行解，但不保证不超过最大值
                        print(f"成功生成问题实例: {num_variables}变量, {num_constraints}约束, 可行解数量>= {min_feasible_count}, 尝试次数={attempts}")
                        return A.tolist(), b.tolist(), c.tolist(), len(feasible_solutions)
    
    # 如果达到最大尝试次数仍未找到合适的实例，抛出异常
    raise Exception(f"在{max_attempts}次尝试后仍未找到满足条件的问题实例")

def save_problem_instance(problem_id, num_variables, num_constraints, A, b, c, num_feasible):
    """
    保存问题实例到指定目录
    
    Args:
        problem_id: 问题ID
        num_variables: 变量数量
        num_constraints: 约束数量
        A: 约束矩阵
        b: 右侧向量
        c: 目标函数系数
        num_feasible: 可行解数量
    """
    # 构建目录路径: /home/zhenyusen/qaoa_results/变量数/约束数/问题名/
    base_dir = '/home/zhenyusen/qaoa_results'
    variables_dir = os.path.join(base_dir, str(num_variables))
    constraints_dir = os.path.join(variables_dir, str(num_constraints))
    problem_name = f'problem_instance_{problem_id}'
    problem_dir = os.path.join(constraints_dir, problem_name)
    
    # 创建目录（如果不存在）
    os.makedirs(problem_dir, exist_ok=True)
    
    # 保存问题参数到problem_params.txt
    params_file = os.path.join(problem_dir, 'problem_params.txt')
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write(f"问题编号: {problem_id}\n")
        f.write(f"变量数: {num_variables}\n")
        f.write(f"约束数: {num_constraints}\n")
        f.write(f"约束矩阵 A = {A}\n")
        f.write(f"右侧向量 b = {b}\n")
        f.write(f"目标函数系数 c = {c}\n")
        f.write(f"可行解数量: {num_feasible}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 保存为JSON格式，便于程序读取
    json_file = os.path.join(problem_dir, 'problem_params.json')
    problem_data = {
        "A": A,
        "b": b,
        "c": c,
        "num_variables": num_variables,
        "num_constraints": num_constraints
    }
    with open(json_file, 'w') as f:
        json.dump(problem_data, f, indent=2)
    
    return problem_dir

def generate_and_save_problems(num_variables, num_constraints, num_problems=5, min_feasible_count=10, max_feasible_count=70, N=8):
    """
    生成并保存问题实例
    
    Args:
        num_variables: 变量数量
        num_constraints: 约束数量
        num_problems: 要生成的问题数量
        min_feasible_count: 最小可行解数量
        max_feasible_count: 最大可行解数量
        N: 变量的最大值限制
    """
    print(f"开始生成 {num_problems} 个{num_variables}变量{num_constraints}约束的问题实例...")
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 检查已有的实例数量
    base_dir = '/home/zhenyusen/qaoa_results'
    variables_dir = os.path.join(base_dir, str(num_variables))
    constraints_dir = os.path.join(variables_dir, str(num_constraints))
    
    existing_instances = []
    if os.path.exists(constraints_dir):
        for item in os.listdir(constraints_dir):
            if item.startswith("problem_instance_") and os.path.isdir(os.path.join(constraints_dir, item)):
                try:
                    instance_id = int(item.split("_")[-1])
                    existing_instances.append(instance_id)
                except ValueError:
                    pass
    
    # 找到下一个可用的实例ID
    start_id = 1
    if existing_instances:
        start_id = max(existing_instances) + 1
    
    # 生成指定数量的问题
    for i in range(num_problems):
        problem_id = start_id + i
        try:
            print(f"生成问题 #{problem_id}: {num_variables}变量, {num_constraints}约束")
            A, b, c, num_feasible = generate_random_ip_instance(
                num_variables=num_variables,
                num_constraints=num_constraints,
                min_feasible_count=min_feasible_count,
                max_feasible_count=max_feasible_count,
                N=N
            )
            problem_dir = save_problem_instance(problem_id, num_variables, num_constraints, A, b, c, num_feasible)
            print(f"  保存到: {problem_dir}")
        except Exception as e:
            print(f"  生成失败: {str(e)}")
    
    print(f"\n已完成生成 {num_problems} 个问题实例，从 {start_id} 到 {start_id + num_problems - 1}！")

if __name__ == "__main__":
    print("===========================================")
    print("批量生成整数规划问题实例 - 命令行版本")
    print("===========================================")
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成混合整数规划问题实例')
    parser.add_argument('--num_variables', type=int, required=True,
                        help='变量数量')
    parser.add_argument('--num_constraints', type=int, required=True,
                        help='约束数量')
    parser.add_argument('--num_problems', type=int, default=5,
                        help='要生成的问题实例数量，默认为5')
    parser.add_argument('--min_feasible', type=int, default=10,
                        help='最小可行解数量，默认为10')
    parser.add_argument('--max_feasible', type=int, default=70,
                        help='最大可行解数量，默认为70')
    parser.add_argument('--max_value', type=int, default=8,
                        help='变量的最大值限制，默认为8')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 验证输入参数
    if args.num_variables < 2 or args.num_variables > 10:
        print("错误：变量数量应在2到10之间")
        exit(1)
    
    if args.num_constraints < 1 or args.num_constraints > args.num_variables:
        print("错误：约束数量应在1到变量数量之间")
        exit(1)
    
    print(f"参数设置:")
    print(f"  变量数量: {args.num_variables}")
    print(f"  约束数量: {args.num_constraints}")
    print(f"  问题实例数: {args.num_problems}")
    print(f"  可行解数量范围: {args.min_feasible}-{args.max_feasible}")
    print(f"  变量最大值: {args.max_value}")
    print("===========================================")
    
    # 生成问题实例
    generate_and_save_problems(
        num_variables=args.num_variables,
        num_constraints=args.num_constraints,
        num_problems=args.num_problems,
        min_feasible_count=args.min_feasible,
        max_feasible_count=args.max_feasible,
        N=args.max_value
    )