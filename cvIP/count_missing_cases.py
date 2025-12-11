import os
import re
import sys
import zipfile
import shutil
from datetime import datetime

def check_missing_cases(instance_path):
    """检查实例目录中缺少哪些情况"""
    missing_cases = []
    
    # 需要检查的p值和电路类型
    p_values = [1, 2, 3, 4, 5, 6, 7]
    circuit_types = ['beta_gamma', 'multi_beta', 'multi_beta_oneH']
    
    for p in p_values:
        for circuit_type in circuit_types:
            # 检查结果文件是否存在
            result_file = os.path.join(instance_path, f"results_p{p}_{circuit_type}.txt")
            plot_file = os.path.join(instance_path, f"plot_p{p}_{circuit_type}.svg")
            
            if not (os.path.exists(result_file) and os.path.exists(plot_file)):
                missing_cases.append((p, circuit_type))
    
    return missing_cases

def backup_problem_data(base_dir, instance_count, num_variables=None, num_constraints=None):
    """
    将问题数据备份压缩保存到指定目录
    
    Args:
        base_dir: 问题数据基础目录
        instance_count: 实例数量
        num_variables: 变量数（可选）
        num_constraints: 约束数（可选）
    
    Returns:
        str: 备份文件路径
    """
    try:
        # 生成带时间戳、变量数、约束数和实例数的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建文件名部分
        name_parts = ["qaoa_backup", timestamp]
        if num_variables is not None:
            name_parts.append(f"vars{num_variables}")
        if num_constraints is not None:
            name_parts.append(f"cons{num_constraints}")
        name_parts.append(f"instances{instance_count}")
        
        backup_filename = "_" .join(name_parts) + ".zip"
        backup_path = os.path.join("/home/zhenyusen", backup_filename)
        
        print(f"\n开始备份问题数据到: {backup_path}")
        print(f"备份源目录: {base_dir}")
        
        # 创建临时目录存储要备份的文件信息
        temp_info_file = os.path.join("/tmp", f"qaoa_backup_info_{timestamp}.txt")
        
        with open(temp_info_file, 'w') as f:
            f.write(f"QAOA问题数据备份\n")
            f.write(f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实例总数: {instance_count}\n")
            f.write(f"备份源目录: {base_dir}\n\n")
            
            # 写入每个实例的信息
            for item in os.listdir(base_dir):
                if item.startswith('problem_instance_'):
                    instance_path = os.path.join(base_dir, item)
                    if os.path.isdir(instance_path):
                        f.write(f"实例目录: {item}\n")
                        # 统计该实例下的文件数
                        file_count = 0
                        for root, _, files in os.walk(instance_path):
                            file_count += len(files)
                        f.write(f"  文件数量: {file_count}\n")
        
        # 创建压缩文件
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加临时信息文件
            zipf.write(temp_info_file, os.path.basename(temp_info_file))
            
            # 遍历并添加所有实例目录下的文件
            for root, _, files in os.walk(base_dir):
                # 跳过__pycache__等不需要的目录
                if '__pycache__' in root or '.git' in root:
                    continue
                
                for file in files:
                    # 只备份结果文件和参数文件
                    if file.endswith('.txt') or file.endswith('.svg') or file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_dir)
                        zipf.write(file_path, arcname)
        
        # 删除临时信息文件
        if os.path.exists(temp_info_file):
            os.remove(temp_info_file)
        
        # 获取备份文件大小
        backup_size = os.path.getsize(backup_path) / (1024 * 1024)  # MB
        
        print(f"备份完成！")
        print(f"备份文件路径: {backup_path}")
        print(f"备份文件大小: {backup_size:.2f} MB")
        print(f"备份文件包含: 信息文件 + 所有结果文件、参数文件和图表文件")
        
        return backup_path
    except Exception as e:
        print(f"备份失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def count_all_missing_cases(base_dir, perform_backup=True, num_variables=None, num_constraints=None):
    """统计所有实例中缺失的案例总数
    
    Args:
        base_dir: 问题数据基础目录
        perform_backup: 是否执行备份
        num_variables: 指定变量数（可选）
        num_constraints: 指定约束数（可选）
    """
    total_missing = 0
    instance_missing = {}
    completed_instances = []
    
    # 获取所有实例目录
    instance_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith('problem_instance_'):
            instance_path = os.path.join(base_dir, item)
            if os.path.isdir(instance_path):
                # 提取实例编号
                match = re.search(r'problem_instance_(\d+)', item)
                if match:
                    instance_id = int(match.group(1))
                    
                    # 如果指定了变量数和约束数，检查是否匹配
                    if num_variables is not None or num_constraints is not None:
                        # 读取问题参数文件
                        param_file = os.path.join(instance_path, 'problem_params.txt')
                        if os.path.exists(param_file):
                            try:
                                with open(param_file, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    instance_vars = None
                                    instance_cons = None
                                    
                                    for line in lines:
                                        if line.startswith('变量数:'):
                                            instance_vars = int(line.split(':')[1].strip())
                                        elif line.startswith('约束数:'):
                                            instance_cons = int(line.split(':')[1].strip())
                                    
                                    # 检查是否匹配指定的变量数和约束数
                                    if ((num_variables is None or instance_vars == num_variables) and 
                                        (num_constraints is None or instance_cons == num_constraints)):
                                        instance_dirs.append((instance_path, instance_id))
                            except Exception:
                                # 如果读取参数失败，跳过该实例
                                continue
                    else:
                        # 没有指定变量数和约束数，添加所有实例
                        instance_dirs.append((instance_path, instance_id))
    
    # 按实例编号排序
    instance_dirs.sort(key=lambda x: x[1])
    
    print(f"找到 {len(instance_dirs)} 个问题实例目录\n")
    print("实例ID列表:", [id for _, id in instance_dirs])
    
    # 统计每个实例的缺失情况
    for instance_path, instance_id in instance_dirs:
        missing_cases = check_missing_cases(instance_path)
        count = len(missing_cases)
        total_missing += count
        
        if count > 0:
            instance_missing[instance_id] = missing_cases
            print(f"问题实例 {instance_id}: 缺失 {count} 个案例")
            for p, circuit_type in missing_cases:
                print(f"  - p={p}, circuit_type={circuit_type}")
        else:
            print(f"问题实例 {instance_id}: 所有案例已完成")
            completed_instances.append(instance_id)
    
    print(f"\n=======================================")
    print(f"总计: {total_missing} 个案例尚未完成")
    print(f"实例总数: {len(instance_dirs)}")
    print(f"已完成实例: {len(completed_instances)} ({completed_instances})")
    print(f"未完成实例: {len(instance_missing)} ({list(instance_missing.keys())})")
    print("=======================================")
    
    # 根据参数决定是否执行备份
    backup_path = None
    if perform_backup:
        backup_path = backup_problem_data(base_dir, len(instance_dirs), num_variables, num_constraints)
    else:
        print("\n已跳过备份操作（仅查看模式）")
    
    return len(completed_instances), completed_instances, backup_path

if __name__ == "__main__":
    print("开始统计缺失的QAOA案例...")
    
    # 解析命令行参数
    perform_backup = True
    num_variables = None
    num_constraints = None
    
    # 处理命令行参数
    # 支持格式: python count_missing_cases.py [1] [variables] [constraints]
    # 1表示仅查看模式，不备份
    if len(sys.argv) > 1:
        arg_index = 1
        # 检查是否为仅查看模式
        if sys.argv[arg_index] == '1':
            perform_backup = False
            print("运行模式: 仅查看统计结果（不执行备份）")
            arg_index += 1
        
        # 检查是否指定了变量数
        if arg_index < len(sys.argv):
            try:
                num_variables = int(sys.argv[arg_index])
                print(f"指定变量数: {num_variables}")
                arg_index += 1
            except ValueError:
                pass
        
        # 检查是否指定了约束数
        if arg_index < len(sys.argv):
            try:
                num_constraints = int(sys.argv[arg_index])
                print(f"指定约束数: {num_constraints}")
            except ValueError:
                pass
    
    # 基础目录
    if num_variables is not None and num_constraints is not None:
        # 如果指定了变量数和约束数，自动定位到对应的目录
        base_dir = f'/home/zhenyusen/qaoa_results/{num_variables}/{num_constraints}'
    else:
        base_dir = '/home/zhenyusen/qaoa_results/4/4'
    
    print(f"检查目录: {base_dir}")
    # 确保目录存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
        sys.exit(1)
    
    # 列出目录内容以验证
    print("目录内容:")
    filtered_items = []
    for item in os.listdir(base_dir):
        if item.startswith('problem_instance_'):
            # 如果指定了变量数和约束数，只显示匹配的实例
            if num_variables is not None or num_constraints is not None:
                instance_path = os.path.join(base_dir, item)
                param_file = os.path.join(instance_path, 'problem_params.txt')
                if os.path.exists(param_file):
                    try:
                        with open(param_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            instance_vars = None
                            instance_cons = None
                            
                            for line in lines:
                                if line.startswith('变量数:'):
                                    instance_vars = int(line.split(':')[1].strip())
                                elif line.startswith('约束数:'):
                                    instance_cons = int(line.split(':')[1].strip())
                            
                            if ((num_variables is None or instance_vars == num_variables) and 
                                (num_constraints is None or instance_cons == num_constraints)):
                                filtered_items.append(item)
                    except Exception:
                        continue
            else:
                filtered_items.append(item)
    
    # 显示过滤后的实例列表
    if filtered_items:
        for item in filtered_items:
            print(f"  {item}")
    else:
        print("  未找到匹配的实例")
    
    # 开始统计
    print("\n开始详细统计...")
    completed_count, completed_instances, backup_path = count_all_missing_cases(
        base_dir, 
        perform_backup=perform_backup,
        num_variables=num_variables,
        num_constraints=num_constraints
    )
    
    print("\n统计任务完成！")
    if backup_path:
        print(f"备份文件已保存至: {backup_path}")
    print("\n使用方法:")
    print("  python count_missing_cases.py                       # 统计并执行备份（默认4变量4约束）")
    print("  python count_missing_cases.py 1                    # 仅查看统计结果（不备份，默认4变量4约束）")
    print("  python count_missing_cases.py <变量数> <约束数>      # 统计指定变量数和约束数的问题并备份")
    print("  python ./mip/count_missing_cases.py 1 <变量数> <约束数>   # 仅查看指定变量数和约束数的问题统计结果（不备份）")