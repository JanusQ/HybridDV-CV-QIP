
import time
import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply, LinearOperator
import os
import json
class BosonicKhosraviVQASolver:
    """
    Variational (QAOA-like) implementation of the Khosravi et al. integer programming algorithm.
    Optimized with expm_multiply for faster state evolution.
    """

    def __init__(
        self,
        A: np.ndarray, 
        b: np.ndarray,
        c: np.ndarray,
        N: int = 7,
        p: int = 3,  # Number of layers
        lambda_penalty: float = 10.0,
        p0: float = 0.55,  # Mixing parameter from paper
        r_squeeze: float = 0.5, # Squeezing from paper
        maxiter: int = 1000,
        seed: int = 42,
        problem_name: str = "khosravi_vqa"
    ):
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.c = np.array(c, dtype=float)
        self.N = N
        self.p = p
        self.lambda_penalty = lambda_penalty
        self.p0 = p0
        self.r_squeeze = r_squeeze
        self.maxiter = maxiter
        self.seed = seed
        self.problem_name = problem_name
        self.num_modes = self.c.shape[0]

        np.random.seed(seed)
        
        # 1. Build Operators
        self.a_ops, self.ad_ops, self.n_ops = self._build_operators()
        self.p_ops = self._build_momentum_operators()
        
        # 2. Build Hamiltonians
        self.H_P = self._build_problem_hamiltonian()
        self.H_M = self._build_mixing_hamiltonian()
        
        # 3. Initial State
        self.initial_state = self._build_initial_state()
        
        # Pre-compute feasible states for analysis
        self.feasible_states = self._find_feasible_states()
        if self.feasible_states:
            self.optimal_state = max(self.feasible_states, key=lambda ns: sum(self.c * np.array(ns)))
            self.optimal_value = sum(self.c * np.array(self.optimal_state))
        else:
            self.optimal_state = None
            self.optimal_value = -np.inf

    def _build_operators(self):
        a_ops, ad_ops, n_ops = [], [], []
        for i in range(self.num_modes):
            ops = [qt.qeye(self.N)] * self.num_modes
            ops[i] = qt.destroy(self.N)
            a = qt.tensor(ops)
            a_ops.append(a)
            ad_ops.append(a.dag())
            n_ops.append(a.dag() * a)
        return a_ops, ad_ops, n_ops

    def _build_momentum_operators(self):
        # p = (a - a_dag) / (i * sqrt(2))
        p_ops = []
        for i in range(self.num_modes):
            p = (self.a_ops[i] - self.ad_ops[i]) / (1j * np.sqrt(2))
            p_ops.append(p)
        return p_ops

    def _build_problem_hamiltonian(self) -> qt.Qobj:
        """Constructs H_P according to Eq. 5."""
        H_obj = sum(-self.c[i] * self.n_ops[i] for i in range(self.num_modes))
        H_penalty = 0
        for j in range(self.A.shape[0]):
            constraint_term = sum(self.A[j, i] * self.n_ops[i] for i in range(self.num_modes)) - self.b[j]
            H_penalty += self.lambda_penalty * (constraint_term ** 2)
        return H_obj + H_penalty

    def _build_mixing_hamiltonian(self) -> qt.Qobj:
        """Constructs H_M according to Eq. 3."""
        H_M = 0
        for i in range(self.num_modes):
            term = self.p_ops[i] - self.p0
            H_M += term ** 2
        return H_M

    def _build_initial_state(self) -> qt.Qobj:
        alpha_val = 1j * self.p0 / np.sqrt(2)
        states = []
        for i in range(self.num_modes):
            psi = qt.squeeze(self.N, -self.r_squeeze) * qt.basis(self.N, 0)
            psi = qt.displace(self.N, alpha_val) * psi
            states.append(psi)
        return qt.tensor(states)

    def _find_feasible_states(self):
        feasible = []
        for ns in product(range(self.N), repeat=self.num_modes):
            if np.allclose(self.A @ np.array(ns), self.b):
                feasible.append(tuple(ns))
        return feasible

    def _apply(self, H_qobj, theta, state):
        """
        Efficiently applies exp(-i * theta * H) * state using sparse matrix vector multiplication.
        """
        # QuTiP state to flattened numpy array (Fortran order matches QuTiP internal storage)
        v = state.full().ravel(order='F')
        D = v.size
        
        # Access sparse data directly
        # Note: QuTiP Qobj.data is usually CSR. Converting to DIA as requested, 
        # though CSR is also fine for dot products.
        H_sparse = H_qobj.data_as(format='dia_matrix')

        def matvec(x):
            return (-1j * theta) * H_sparse.dot(x)

        def rmatvec(x):
            return (1j * theta) * H_sparse.conj().T.dot(x)

        Aop = LinearOperator((D, D), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)

        # Trace calculation for traceA (helps numerical stability in expm_multiply)
        try:
            # H_M typically has trace > 0, H_P depends on problem
            trH = H_qobj.tr()
        except Exception:
            trH = 0.0
        traceA = (-1j * theta) * trH

        # Compute w = exp(Aop) * v
        w = expm_multiply(Aop, v, traceA=traceA)
        
        # Reshape back to QuTiP Qobj
        return qt.Qobj(w.reshape(state.shape, order='F'), dims=state.dims)

    def variational_circuit(self, params: np.ndarray) -> qt.Qobj:
        """
        Apply layers: U = ... e^{-i beta H_M} e^{-i gamma H_P} |psi_0>
        """
        state = self.initial_state.copy()
        
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            # Use optimized _apply instead of .expm()
            state = self._apply(self.H_P, gamma, state)
            state = self._apply(self.H_M, beta, state)
            
        return state

    def optimize(self):
        """Find optimal gamma/beta parameters to minimize <H_P>."""
        self.history = {"cost": [], "params": []}
        
        def cost_fn(params):
            final_state = self.variational_circuit(params)
            cost = qt.expect(self.H_P, final_state)
            self.history["cost"].append(cost)
            self.history["params"].append(params)
            print(cost)
            return cost

        init_params = np.random.uniform(0, 0.5, 2 * self.p)
        
        print(f"Starting VQA Optimization ({self.p} layers) using expm_multiply...")
        start_t = time.time()
        
        res = minimize(
            cost_fn, 
            init_params, 
            method='COBYLA', 
            options={'maxiter': self.maxiter, 'disp': True}
        )
        
        end_t = time.time()
        print(f"Optimization done in {end_t - start_t:.2f}s. Final Cost: {res.fun:.4f}")
        
        self.optimal_params = res.x
        self.final_state = self.variational_circuit(self.optimal_params)
        
        return self._analyze_results(res.fun)

    def _analyze_results(self, final_cost):
        data = self.final_state.data.to_array().flatten()
        dims = [self.N] * self.num_modes
        
        feasible_prob_sum = 0
        optimal_state_prob = 0
        max_prob = -1.0
        max_state_tuple = None
        
        for idx in range(len(data)):
            amp = data[idx]
            prob = abs(amp)**2
            ns = np.unravel_index(idx, dims)
            ns = tuple(int(x) for x in ns)
            
            if prob > max_prob:
                max_prob = prob
                max_state_tuple = ns
            
            if ns in self.feasible_states:
                feasible_prob_sum += prob
                if self.optimal_state and ns == self.optimal_state:
                    optimal_state_prob = prob
                    
        return {
            "final_cost": final_cost,
            "optimal_params": self.optimal_params,
            "feasible_prob_sum": feasible_prob_sum,
            "optimal_state_prob": optimal_state_prob,
            "most_likely_state": max_state_tuple,
            "most_likely_prob": max_prob,
            "target_optimal_state": self.optimal_state
        }

    # ... (Include get_problem_metrics, get_solution_metrics, save_results_to_file from previous context) ...
    def get_problem_metrics(self) -> dict:
        m, d = self.A.shape
        opt_val = self.optimal_value if self.optimal_value != -np.inf else 0.0
        return {
            "problem_name": self.problem_name,
            "parameters": {"N": self.N, "p": self.p, "lambda": self.lambda_penalty},
            "num_variables": d,
            "num_constraints": m,
            "optimal_cost": opt_val
        }

    def get_solution_metrics(self) -> dict:
        final_objective = self.history["cost"][-1]
        
        feasible_prob_sum = 0.0
        optimal_state_prob = 0.0
        for ns in self.feasible_states:
            basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
            prob = abs(basis.overlap(self.final_state)) ** 2
            feasible_prob_sum += prob
            if self.optimal_state and ns == self.optimal_state:
                optimal_state_prob = prob
        
        opt_val = self.optimal_value
        ARG = abs(1- final_objective/opt_val)  if opt_val not in [0, -np.inf] else 0.0

        return {
            "method": "Khosravi VQA",
            "feasible_probability_sum": feasible_prob_sum,
            "cost_history": self.history["cost"],
            "ARG": ARG,
            "success_rate": optimal_state_prob,
            "final_cost": self.history["cost"][-1]
        }

    def save_results_to_file(self, filename: str = "qaoa_results.txt", A=None, b=None, c=None) -> None:
        import os
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        problem_metrics = self.get_problem_metrics()
        solution_metrics = self.get_solution_metrics()
        problem_metrics.update(solution_metrics)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problem_metrics,f)

    def plot_results(self, save_path="khosravi_vqa_results.svg"):
        res = self._analyze_results(self.history["cost"][-1])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history["cost"], marker='.')
        axes[0].set_title(f"Convergence (p={self.p})")
        axes[0].set_ylabel("<H_P>")
        
        if self.feasible_states:
            feas_probs = []
            labels = []
            for ns in self.feasible_states:
                basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
                prob = abs(basis.overlap(self.final_state)) ** 2
                feas_probs.append(prob)
                labels.append(str(ns))
            
            sorted_indices = np.argsort(feas_probs)[::-1][:10]
            top_probs = [feas_probs[i] for i in sorted_indices]
            top_labels = [labels[i] for i in sorted_indices]
            
            axes[1].bar(range(len(top_probs)), top_probs)
            axes[1].set_xticks(range(len(top_probs)))
            axes[1].set_xticklabels(top_labels, rotation=45, ha='right')
            axes[1].set_title("Top Feasible States")
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

def generate_random_ip_instance(num_variables, num_constraints, max_coeff=3, max_rhs=30, N=10, feasible_count=10): 
    """ 
    随机生成整数规划问题实例：A x = b 
    
    Args: 
        num_variables: 变量数量 (d) 
        num_constraints: 约束数量 (m) 
        max_coeff: 系数矩阵A的最大绝对值 
        max_rhs: 右侧向量b的最大绝对值 
        N: 可行解的最大值限制
        feasible_count: 要求的最小可行解数量
    
    Returns: 
        A, b, c, num_feasible: 约束矩阵、右侧向量、目标函数系数、可行解数量 
    """
    
    while True:
        # 生成随机系数矩阵A (整数)
        A = np.random.randint(1, max_coeff + 1, size=(num_constraints, num_variables)) 
        
        # 生成随机可行解x0，确保x0中的每个元素都小于N
        x0 = np.random.randint(1, min(N, 5), size=num_variables) 
        
        # 计算b = A x0，确保存在可行解
        b = A @ x0 
        
        # 生成随机目标函数系数c
        c = np.random.randint(1, max_coeff + 1, size=num_variables) 
        
        # 检查是否有足够的可行解
        feasible_solutions = []
        # 遍历所有可能的解（简化版本，实际问题中可能需要更高效的方法）
        # 注意：当变量数量较多时，这种方法会非常慢，需要优化
        if num_variables <= 4:  # 只在变量数量较少时检查可行解数量
            for x_candidate in product(range(N), repeat=num_variables):
                if np.allclose(A @ np.array(x_candidate), b):
                    feasible_solutions.append(x_candidate)
                    if len(feasible_solutions) >= feasible_count:
                        break
        
        # 如果找到足够的可行解，或者变量数量太多无法快速检查，则返回当前实例
        if len(feasible_solutions) >= feasible_count or num_variables > 4:
            break
    
    # 计算可行解数量
    num_feasible = 0
    # 仅在变量数量较少时计算精确的可行解数量
    if num_variables <= 6:  # 6个变量在N=7时大约有117,649种组合，这是可接受的
        for x_candidate in product(range(N), repeat=num_variables):
            if np.allclose(A @ np.array(x_candidate), b):
                num_feasible += 1
    else:
        # 对于变量数量较多的情况，至少确认找到的可行解数量
        num_feasible = len(feasible_solutions)
    
    return A.tolist(), b.tolist(), c.tolist()

# Example usage for original problem
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    seed = 42
    np.random.seed(seed)
    
    # 默认的变量数和约束数
    # 这些值可能会被命令行参数覆盖
    num_variables = 4      # 变量数量 d
    num_constraints = 1    # 约束数量 m
    set_N =7
    
    # 支持三种电路类型
    # 从命令行参数获取运行参数（用于并行执行）
    import sys
    import os
    import argparse
    run_id = 1
    max_iter = 70
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行QAOA求解器')
    parser.add_argument('run_id', type=int, nargs='?', default=run_id, help='运行ID')
    parser.add_argument('--num_variables', type=int, default=num_variables, help='变量数量')
    parser.add_argument('--num_constraints', type=int, default=num_constraints, help='约束数量')
    parser.add_argument('--max_iter', type=int, default=max_iter, help='迭代次数')
    # 如果提供了位置参数，使用它们
    # 这是为了保持向后兼容性
    if len(sys.argv) > 1:
        # 解析位置参数
        args, unknown = parser.parse_known_args()
        run_id = args.run_id
        
        # 覆盖默认的变量数和约束数
        num_variables = args.num_variables
        num_constraints = args.num_constraints
        max_iter = args.max_iter
    
    # 设置随机种子以确保不同运行ID产生不同结果
    np.random.seed(seed + run_id)
    

    A, b, c = generate_random_ip_instance(num_variables, num_constraints)
    
    # 创建问题特定的文件夹（使用默认编号0）
    problem_dir = f"vars{num_variables}/cons{num_constraints}/problem_instance_{run_id}"
    os.makedirs(problem_dir, exist_ok=True)
    
    # 保存问题参数到单独文件
    with open(os.path.join(problem_dir, "problem_params.txt"), 'w', encoding='utf-8') as f:
        f.write(f"问题编号: 0\n")
        f.write(f"变量数: {num_variables}\n")
        f.write(f"约束数: {num_constraints}\n")
        f.write(f"约束矩阵 A = {A}\n")
        f.write(f"右侧向量 b = {b}\n")
        f.write(f"目标函数系数 c = {c}\n")
    
    print(f"\n随机生成的整数规划实例：")
    print(f"变量数: {num_variables}, 约束数: {num_constraints}")
    print(f"约束矩阵 A = {A}")
    print(f"右侧向量 b = {b}")
    print(f"目标函数系数 c = {c}")
    print(f"结果保存目录: {problem_dir}")
    
    for p in range(1, 8):
        vqa_solver = BosonicKhosraviVQASolver(
            A, b, c,
            N=set_N,
            p=p,  # Default to 4 layers if not specified
            lambda_penalty=4.0,  # [cite: 210]
            p0=0.55 * np.pi,     # [cite: 377] (Using similar scaling)
            r_squeeze=0.8,       # [cite: 156]
            maxiter= max_iter,
            problem_name="khosravi_vqa_run"
        )
        problem_name = f"khosravi_vqa_run_p{p}"
        # Run Optimization
        metrics = vqa_solver.optimize()
        result_file = os.path.join(problem_dir,f"khosravi_vqa_results_p{p}.json")
        vqa_solver.save_results_to_file(result_file, A, b, c)
        # Report
        print(f"\n--- VQA Results ---")
        print(f"Final Cost <H_P>: {metrics['final_cost']:.4f}")
        print(f"Optimal State (Target): {metrics['target_optimal_state']}")
        print(f"Most Likely State Found: {metrics['most_likely_state']} (P={metrics['most_likely_prob']:.4f})")
        print(f"Total Probability in Feasible Subspace: {metrics['feasible_prob_sum']:.4f}")
        print(f"Probability of Optimal State: {metrics['optimal_state_prob']:.4f}")
        print(f"Optimal Parameters: {metrics['optimal_params']}")
        
        plot_path = os.path.join(problem_dir, f"plot_{problem_name}.svg")
        vqa_solver.plot_results(save_path=plot_path)
        