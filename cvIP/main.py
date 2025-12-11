import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply, LinearOperator
import sympy as sp
from typing import List, Tuple, Optional, Union, Dict
from scipy.sparse import csr_matrix
import time

class BosonicQAOAIPSolver:
    """
    Bosonic QAOA solver for arbitrary non-negative integer linear programming (IP):
    max c^T x s.t. A x = b, x >= 0 integer.
    
    Encoding: x_i <-> n_i (photon number in mode i).
    Constraint subspace S_c: { |n> | A n = b }.
    Driver H_M: sum_u (O_u + O_u^dagger) over integer nullspace basis of A.
    Cost H_C = - sum c_i n_i.
    
    Args:
        A: Constraint matrix (m x d, integer coeffs).
        b: RHS vector (m, integer targets).
        c: Objective coeffs (d, maximize c^T x). 线性目标函数 length(c) = num_variables
        N: Fock truncation per mode.
        p: QAOA layers.
        num_modes: d (inferred from A/c if not given). num_variables
        g: Driver coupling strength.
        maxiter: Optimization iterations.
        seed: Random seed.
    """
    
    def __init__(
        self,
        A: np.ndarray, 
        b: np.ndarray,
        c: np.ndarray,
        N: int = 7,
        p: int = 2,
        g: float = 1.0,
        maxiter: int = 500,
        seed: int = 42,
        num_modes: Optional[int] = None,
        circuit_type: str = "beta_gamma",  # or "multi_beta"
        problem_name: str = "default_problem"  # 添加问题名称参数
    ):
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.c = np.array(c, dtype=float)
        self.N = N # 变量的大小，取值范围 [0, N - 1]
        self.p = p
        self.g = g
        self.maxiter = maxiter
        self.seed = seed
        self.num_modes = num_modes or self.c.shape[0] or self.A.shape[1] # 变量个数
        self.problem_name = problem_name  # 保存问题名称
        
        # Validate dimensions
        if self.A.shape[1] != self.num_modes or self.c.shape[0] != self.num_modes:
            raise ValueError("Dimensions mismatch: A (m x d), b (m), c (d).")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A rows != b length.")
        
        np.random.seed(seed)
        
        # Compute integer nullspace basis using sympy
        self.null_space_basis = self._compute_integer_nullspace()
        print(f"Computed integer nullspace basis with {len(self.null_space_basis)} vectors.")
        for null_vec in self.null_space_basis:
            print(f"  Null vector: {null_vec}")
        # Build Hamiltonians and operators
        self.a_ops, self.ad_ops, self.n_ops = self._build_operators()
        self.H_C = self._build_cost_hamiltonian()
        self.H_ds = self._build_seperate_driver_hamiltonian()
        self.H_M = sum(self.H_ds)
        self.target_operator = -self.H_C  # For expectation: max c^T n
        self.circuit_type = circuit_type
        if circuit_type == "beta_gamma":
            self.qaoa_circuit = self.qaoa_circuit_beta_gamma
            self.paramlength = 2  # gamma, beta per layer
        elif circuit_type == "multi_beta":
            self.qaoa_circuit = self.qaoa_multi_beta_layer_circuit
            self.paramlength = len(self.null_space_basis)  # beta_i per layer
        elif circuit_type == "multi_beta_oneH":
            self.qaoa_circuit = self.qaoa_multi_beta_oneH_circuit
            self.paramlength = len(self.null_space_basis)  # beta_i per layer
        else:
            raise ValueError("circuit_type must be 'beta_gamma' or 'multi_beta'")
        # Find feasible states for initial and tracking
        self.feasible_states = self._find_feasible_states()
        if not self.feasible_states:
            raise ValueError("No feasible states found in truncation N.")
        
        # Tracked states: sorted by objective descending (top 6 or all if fewer)
        self.initial_state = min(self.feasible_states, key=lambda ns: sum(self.c * ns))
        # self.initial_state = np.array([2,0,3,3])
        self.tracked_states = sorted(self.feasible_states, key=lambda ns: sum(self.c * ns), reverse=True)[:6]+[self.initial_state]
        self.tracked_labels = [f"|{','.join(map(str, ns))}⟩ (obj={sum(self.c * ns):.1f})" for ns in self.tracked_states]
        
        # 计算最优成本 (最优可行解的目标函数值)
        self.optimal_cost = max(sum(self.c * np.array(ns)) for ns in self.feasible_states)
        self.optimal_state = max(self.feasible_states, key=lambda ns: sum(self.c * np.array(ns)))
        
        print(f"Initialized BosonicQAOAIPSolver: {self.num_modes} modes, {len(self.null_space_basis)} null vectors.")
        print(f"Constraints: A x = b (shape {self.A.shape}). Objective: max {self.c} · x.")
        print(f"Feasible subspace dim: {len(self.feasible_states)} (in truncation N={N}).")
        print(f"initial_state: |{','.join(map(str, self.initial_state))}⟩")
        print(f"Optimal cost: {self.optimal_cost:.4f} at state |{','.join(map(str, self.optimal_state))}⟩")
    
    def _compute_integer_nullspace(self) -> List[np.ndarray]:
        """Compute primitive integer basis for ker(A) using sympy nullspace."""
        A_sym = sp.Matrix(self.A)
        ns_rational = A_sym.nullspace()
        if not ns_rational:
            raise ValueError("Nullspace empty; constraints overconstrained.")
        
        # Collect denominators
        denoms = []
        for vec in ns_rational:
            for entry in vec:
                if hasattr(entry, 'is_Rational') and entry.is_Rational:
                    denoms.append(entry.q)  # denominator
        lcm_den = sp.lcm(denoms) if denoms else sp.Integer(1)
        
        # Integer vectors
        int_vecs = []
        for vec in ns_rational:
            int_vec = (lcm_den * vec).applyfunc(lambda x: int(x))
            # Make primitive: gcd of components
            int_vec = np.array(int_vec).flatten().astype(int)
            gcd = np.gcd.reduce(int_vec)
            if gcd != 0:
                int_vec = int_vec // gcd
            # Flip sign if first non-zero is negative
            if int_vec[int(np.nonzero(int_vec)[0][0])] < 0:
                int_vec = -int_vec
            int_vecs.append(int_vec)
        
        # Remove duplicates (if any)
        unique_vecs = set(tuple(v) for v in int_vecs)
        unique_vecs = np.array(list(unique_vecs))
        
        # add a new vector for circle loop
        # add first vector and last vector to make a circle loop， then gcd to make it primitive
        if len(unique_vecs)>1:
            first_vec = unique_vecs[0]
            last_vec = unique_vecs[-1]
            new_vec = first_vec + last_vec
            new_vec_minus = first_vec - last_vec
            if len(np.nonzero(new_vec)[0]) < len(np.nonzero(new_vec_minus)[0]):
                new_vec = new_vec
            else:
                new_vec = new_vec_minus
            gcd = np.gcd.reduce(new_vec)
            if gcd != 0:
                new_vec = new_vec // gcd
            if new_vec[int(np.nonzero(new_vec)[0][0])] < 0:
                new_vec = -new_vec
            unique_vecs = np.vstack([unique_vecs, new_vec])
        
        ## add reverse direction
        # reversed_vecs = -unique_vecs
        # unique_vecs = np.vstack([unique_vecs, reversed_vecs])
        return unique_vecs
    
    def _build_operators(self) -> Tuple[List[qt.Qobj], List[qt.Qobj], List[qt.Qobj]]:
        """Build a_i, a_i^dagger, n_i for all modes."""
        a_ops = []
        ad_ops = []
        n_ops = []
        for i in range(self.num_modes):
            ops_list = [qt.qeye(self.N)] * self.num_modes
            ops_list[i] = qt.destroy(self.N)
            a = qt.tensor(ops_list)
            ad = a.dag()
            n = ad * a
            a_ops.append(a)
            ad_ops.append(ad)
            n_ops.append(n)
        return a_ops, ad_ops, n_ops
    
    def _build_cost_hamiltonian(self) -> qt.Qobj:
        """H_C = - sum c_i n_i."""
        H_C = sum(-self.c[i] * self.n_ops[i] for i in range(self.num_modes))
        return H_C
    
    def _build_driver_hamiltonian(self) -> qt.Qobj:
        """H_M = g sum_u (O_u + O_u^dagger)."""
        H_M = 0 * self.a_ops[0]
        for u in self.null_space_basis:
            O_u = qt.qeye(1)
            for i in range(self.num_modes):
                if u[i] > 0:
                    O_u = O_u * (self.ad_ops[i] ** u[i])
                elif u[i] < 0:
                    O_u = O_u * (self.a_ops[i] ** abs(u[i]))
            H_M += self.g * (O_u + O_u.dag())
        return H_M
    
    def _build_seperate_driver_hamiltonian(self) -> List[qt.Qobj]:
        """H_M = g sum_u (O_u + O_u^dagger)."""
        Hds = []
        latex_labels = []
        idx = 1
        for u in self.null_space_basis:
            ## print the latex code for each driver hamiltonian
            latex_label = f"g_{{{idx}}} ("
            for i in range(self.num_modes):
                if u[i] > 0:
                    latex_label += f"a_{{{i}}}^{{{u[i]}}} " if u[i] > 1 else f"a_{{{i}}} "
                elif u[i] < 0:
                    latex_label += f"a_{{{i}}}^{{\\dagger {abs(u[i])}}} " if abs(u[i]) > 1 else f"a_{{{i}}}^{{\\dagger}} "
            ## dagger part
            latex_label += " + "
            for i in range(self.num_modes):
                if u[i] > 0:
                    latex_label += f"a_{{{i}}}^{{\\dagger {u[i]}}} " if u[i] > 1 else f"a_{{{i}}}^{{\\dagger}} "
                elif u[i] < 0:
                    latex_label += f"a_{{{i}}}^{{{abs(u[i])}}} " if abs(u[i]) > 1 else f"a_{{{i}}} "
            latex_label += ")"
            idx += 1
            latex_labels.append(latex_label)
            H_M = 0 * self.a_ops[0]
            O_u = 1
            for i in range(self.num_modes):
                if u[i] > 0:
                    O_u = O_u * (self.ad_ops[i] ** int(u[i]))
                elif u[i] < 0:
                    O_u = O_u * (self.a_ops[i] ** abs(u[i]))
            H_M += self.g * (O_u + O_u.dag())
            Hds.append(H_M)
        self.latex_label_H_M = " + ".join(latex_labels)
        return Hds
    
    def _find_feasible_states(self) -> List[Tuple[int, ...]]:
        # 4 0-6
        """Enumerate feasible n in [0,N)^d with A n = b (integer)."""
        feasible = []
        # N 7 num_modes 
        for ns_tuple in product(range(self.N), repeat=self.num_modes):
            ns = np.array(ns_tuple)
            if np.allclose(self.A @ ns, self.b):
                feasible.append(tuple(ns))
        return feasible
    
    def create_initial_state(self, superposition: bool = False) -> qt.Qobj:
        """Superposition of argmax/argmin objective feasible states, or max if not superpose."""
        if not superposition:
            return qt.tensor(*[qt.fock(self.N, n) for n in self.initial_state])
        
        max_ns = max(self.feasible_states, key=lambda ns: sum(self.c * ns))
        min_ns = min(self.feasible_states, key=lambda ns: sum(self.c * ns))
        state1 = qt.tensor(*[qt.fock(self.N, n) for n in max_ns])
        state2 = qt.tensor(*[qt.fock(self.N, n) for n in min_ns])
        return (state1 + state2).unit()

    def _apply(self, H_qobj, theta, state):
        v = state.full().ravel(order='F')
        D = v.size
        H_sparse = H_qobj.data_as(format='dia_matrix')
        def matvec(x):
            return (-1j * theta) * H_sparse.dot(x)

        def rmatvec(x):
            return (1j * theta) * H_sparse.conj().T.dot(x)

        Aop = LinearOperator((D, D), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)

        # 计算或直接给出 traceA（driver 可直接设 0）
        try:
            trH = H_qobj.tr()   # QuTiP 快速算迹；H_M 会给 0
        except Exception:
            trH = 0.0
        traceA = (-1j * theta) * trH

        w = expm_multiply(Aop, v, traceA=traceA)
        return qt.Qobj(w.reshape(state.shape, order='F'), dims=state.dims)


    def qaoa_circuit_beta_gamma(self, params: np.ndarray, initial_state: qt.Qobj) -> qt.Qobj:
        """p-layer QAOA: prod (U_M(beta) U_C(gamma))."""
        state = initial_state.copy()
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            state = self._apply(self.H_C, gamma, state)
            state = self._apply(self.H_M, beta, state)
        return state
    def qaoa_multi_beta_layer_circuit(self, params: np.ndarray, initial_state: qt.Qobj) -> qt.Qobj:
        """p-layer QAOA: prod (U_d1(beta1) U_d2(beta2) ...)."""
        betalength = len(self.null_space_basis)
        if len(params) != self.p * (betalength):
            raise ValueError(f"Expected {self.p * (betalength)} params, got {len(params)}.")
            
        state = initial_state.copy()
        for layer in range(self.p):
            for i in range(betalength):
                beta = params[layer * (betalength) + i]
                state = self._apply(self.H_ds[i], beta, state)
        return state
    def qaoa_multi_beta_oneH_circuit(self, params: np.ndarray, initial_state: qt.Qobj) -> qt.Qobj:
        """p-layer QAOA: prod (U_d(beta1,beta2)...)."""
        betalength = len(self.null_space_basis)
        if len(params) != self.p * (betalength):
            raise ValueError(f"Expected {self.p * (betalength)} params, got {len(params)}.")
            
        state = initial_state.copy()
        for layer in range(self.p):
            betas = params[layer * (betalength):(layer + 1) * (betalength)]
            H_d_layer = sum(betas[i] * self.H_ds[i] for i in range(betalength))
            # state = (-1j * H_d_layer).expm() * state
            state = self._apply(H_d_layer, 1, state)
        return state
    
    
    def optimize(self, initial_state: Optional[qt.Qobj] = None) -> dict:
        """COBYLA optimization with history tracking."""
        # SPSA

        iteration_count = 0

        if initial_state is None:
            initial_state = self.create_initial_state()
        
        # History dict
        self.iter_history = {"iter": [], "cost": [], "prob": np.zeros((0, len(self.tracked_states)))}
        
        def cost_with_history(params: np.ndarray) -> float:
            iter_idx = len(self.iter_history["iter"])
            self.iter_history["iter"].append(iter_idx + 1)
            
            final_state = self.qaoa_circuit(params, initial_state)
            cost_val = qt.expect(self.H_C, final_state)
            self.iter_history["cost"].append(cost_val)
            
            # Track probs for selected states
            current_probs = []
            for ns in self.tracked_states:
                basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
                prob = abs(basis.overlap(final_state)) ** 2
                current_probs.append(prob)
            self.iter_history["prob"] = np.vstack([self.iter_history["prob"], current_probs])
            
            return cost_val
        
        # Initial params
        init_params = np.random.uniform(0, np.pi, self.paramlength * self.p)


        def callback_cost(params):
            nonlocal iteration_count
            iteration_count += 1

            final_state = self.qaoa_circuit(params, initial_state)
            cost_val = qt.expect(self.H_C, final_state)

            if iteration_count % 5 == 0:
                print(f"iteration {iteration_count}, result: {cost_val}")

                
        result = minimize(
            fun=cost_with_history,
            x0=init_params,
            method="COBYLA",
            options={"maxiter": self.maxiter, "disp": True}
            ,callback=callback_cost
        )
        
        self.optimal_params = result.x
        self.final_state = self.qaoa_circuit(self.optimal_params, initial_state)
        self.final_cost = result.fun
        self.final_obj = -self.final_cost  # Maximized objective
        
        # 计算约束满足率 (所有可行解的概率和)
        self.final_probs = self.get_fock_probs(self.final_state)
        self.feasible_prob_sum = sum(self.final_probs[ns] for ns in self.feasible_states)
        
        # 计算最优解的概率 (成功率)
        self.optimal_state_prob = self.final_probs[tuple(self.optimal_state)]
        
        # 计算ARG值
        self.ARG = abs((self.final_obj - self.optimal_cost) / self.optimal_cost) if self.optimal_cost != 0 else 0
        
        print(f"\nOptimization complete: Optimal params {self.optimal_params.round(4)}, Obj={self.final_obj:.4f}")
        print(f"Feasible probability sum: {self.feasible_prob_sum:.4f}")
        print(f"Optimal state probability: {self.optimal_state_prob:.4f}")
        print(f"ARG value: {self.ARG:.4f}")
        
        return {"params": self.optimal_params, "obj": self.final_obj, "state": self.final_state}
    
    def get_fock_probs(self, state: qt.Qobj) -> np.ndarray:
        """P(n) for all n in [0,N)^d."""
        probs = np.zeros((self.N,) * self.num_modes)
        for ns in product(range(self.N), repeat=self.num_modes):
            basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
            probs[ns] = abs(basis.overlap(state)) ** 2
        return probs
    
    def plot_results(self, save_path: str = "qaoa_ip_results.svg") -> None:
        """Plot iteration history, costs, probs, objectives."""
        # 检查并创建目录（如果包含路径）
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        final_probs = self.get_fock_probs(self.final_state)
        final_tracked_probs = [final_probs[ns] for ns in self.tracked_states]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Tracked state probs over iterations
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tracked_states)))
        for i, label in enumerate(self.tracked_labels):
            axes[0, 0].plot(self.iter_history["iter"], self.iter_history["prob"][:, i],
                            linewidth=2.5, label=label, color=colors[i], marker="o", markersize=3)
        axes[0, 0].set_xlabel("Optimization Iteration")
        axes[0, 0].set_ylabel("State Probability")
        axes[0, 0].set_title(f"QAOA Iteration: Tracked State Probabilities (p={self.p})")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(alpha=0.3)
        
        # Subplot 2: Cost convergence
        axes[0, 1].plot(self.iter_history["iter"], self.iter_history["cost"],
                        linewidth=3, color="red", marker="s", markersize=4)
        axes[0, 1].set_xlabel("Optimization Iteration")
        axes[0, 1].set_ylabel("<H_C>")
        axes[0, 1].set_title("Cost Function Convergence")
        axes[0, 1].grid(alpha=0.3)
        
        # Subplot 3: Final tracked probs bar
        bars = axes[1, 0].bar(range(len(final_tracked_probs)), final_tracked_probs, color=colors)
        for bar, prob in zip(bars, final_tracked_probs):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{prob:.3f}", ha="center", va="bottom", fontsize=10)
        axes[1, 0].set_xlabel("Tracked States")
        axes[1, 0].set_ylabel("Final Probability")
        axes[1, 0].set_title("Final State Probabilities")
        axes[1, 0].set_xticks(range(len(final_tracked_probs)))
        axes[1, 0].set_xticklabels([lbl.split(" (")[0] for lbl in self.tracked_labels], rotation=45, ha="right")
        axes[1, 0].grid(axis="y", alpha=0.3)
        
        # Subplot 4: Objective over iterations
        obj_history = [-cost for cost in self.iter_history["cost"]]
        max_obj = max([sum(self.c * ns) for ns in self.feasible_states])
        axes[1, 1].plot(self.iter_history["iter"], obj_history,
                        linewidth=3, color="green", marker="^", markersize=4)
        axes[1, 1].axhline(y=max_obj, color="orange", linestyle="--", linewidth=2, label=f"Max Obj ({max_obj})")
        axes[1, 1].set_xlabel("Optimization Iteration")
        axes[1, 1].set_ylabel("Objective <c^T x>")
        axes[1, 1].set_title("Objective Improvement")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
    
    def print_summary(self) -> None:
        """Print optimization summary and top states."""
        print(f"\n=== IP Solution Summary ===")
        print(f"Objective: {self.final_obj:.4f} (max possible: {self.optimal_cost:.4f})")
        print(f"Improvement: {self.final_obj + self.iter_history['cost'][0]:.4f}")
        print(f"Feasible probability sum: {self.feasible_prob_sum:.4f}")
        print(f"Optimal state probability: {self.optimal_state_prob:.4f}")
        print(f"ARG value: {self.ARG:.4f}")
        print(f"Total iterations: {len(self.iter_history['iter'])}")
        
        final_probs = self.final_probs
        top_feasible = sorted(
            [(ns, final_probs[ns], sum(self.c * ns))
             for ns in self.feasible_states if final_probs[ns] > 1e-3],
            key=lambda x: x[1], reverse=True
        )
        print(f"\nTop 3 feasible states by probability:")
        for i, (ns, prob, obj) in enumerate(top_feasible[:3]):
            print(f"  {i+1}: |{','.join(map(str, ns))}⟩ → P={prob:.4f}, Obj={obj:.4f}")
    
    def get_problem_metrics(self) -> dict:
        """计算并返回问题特征指标"""
        # 问题特征指标
        m, d = self.A.shape
        
        # 计算平均每个约束中的变量数
        non_zero_count = np.count_nonzero(self.A)
        avg_vars_per_constraint = non_zero_count / m if m > 0 else 0
        
        # 确定约束类型（线性）
        constraint_type = "linear"  # 本实现中只有线性约束
        
        # 确定目标函数类型（线性）
        objective_type = "linear"  # 本实现中只有线性目标函数
        
        # 计算整数变量信息
        num_integer_vars = d
        
        return {
            "problem_name": self.problem_name,
            "parameters": {
                "N": self.N,
                "p": self.p,
                "g": self.g,
                "maxiter": self.maxiter,
                "seed": self.seed
            },
            "num_variables": d,
            "num_integer_variables": num_integer_vars,
            "num_constraints": m,
            "avg_vars_per_constraint": avg_vars_per_constraint,
            "constraint_type": constraint_type,
            "objective_type": objective_type,
            "num_feasible_solutions": len(self.feasible_states),
            "optimal_cost": self.optimal_cost
        }
    
    def get_solution_metrics(self) -> dict:
        """计算并返回求解结果指标"""
        return {
            "method": "QAOA",
            "circuit_type": self.circuit_type,
            "layers": self.p,
            "feasible_probability_sum": self.feasible_prob_sum,
            "cost_history": self.iter_history["cost"],
            "ARG": self.ARG,
            "success_rate": self.optimal_state_prob,
            "num_iterations": len(self.iter_history["iter"]),
            "final_cost": self.final_cost,
            "final_objective": self.final_obj
        }
    
    def save_results_to_file(self, filename: str = "qaoa_results.txt", A=None, b=None, c=None) -> None:
        """将实验结果保存到txt文件中"""
        # 检查并创建目录（如果包含路径）
        import os
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 计算指标
        problem_metrics = self.get_problem_metrics()
        solution_metrics = self.get_solution_metrics()
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            # 保存问题参数A, b, c（如果提供）
            if A is not None and b is not None and c is not None:
                f.write("# 问题参数\n")
                f.write("A = " + str(A) + "\n")
                f.write("b = " + str(b) + "\n")
                f.write("c = " + str(c) + "\n\n")
            
            f.write("# QAOA 实验结果\n\n")
            
            # 问题特征指标
            f.write("## 问题特征指标\n")
            f.write(f"问题名称: {problem_metrics['problem_name']}\n")
            f.write(f"参数: {problem_metrics['parameters']}\n")
            f.write(f"变量数: {problem_metrics['num_variables']}\n")
            f.write(f"整数变量数: {problem_metrics['num_integer_variables']}\n")
            f.write(f"约束数: {problem_metrics['num_constraints']}\n")
            f.write(f"平均每个约束的变量数: {problem_metrics['avg_vars_per_constraint']:.2f}\n")
            f.write(f"约束类型: {problem_metrics['constraint_type']}\n")
            f.write(f"目标函数类型: {problem_metrics['objective_type']}\n")
            f.write(f"可行解数量: {problem_metrics['num_feasible_solutions']}\n")
            f.write(f"最优成本: {problem_metrics['optimal_cost']:.4f}\n\n")
            
            # 方法指标
            f.write("## 方法指标\n")
            f.write(f"算法: {solution_metrics['method']}\n")
            f.write(f"电路类型: {solution_metrics['circuit_type']}\n")
            f.write(f"层数: {solution_metrics['layers']}\n\n")
            
            # 求解结果指标
            f.write("## 求解结果指标\n")
            f.write(f"约束满足率: {solution_metrics['feasible_probability_sum']:.4f}\n")
            f.write(f"ARG值: {solution_metrics['ARG']:.4f}\n")
            f.write(f"成功率: {solution_metrics['success_rate']:.4f}\n")
            f.write(f"迭代次数: {solution_metrics['num_iterations']}\n")
            f.write(f"最终目标函数值: {solution_metrics['final_objective']:.4f}\n\n")
            
            # cost迭代记录
            f.write("## Cost迭代记录\n")
            for i, cost in enumerate(solution_metrics['cost_history']):
                f.write(f"迭代{i+1}: {cost:.6f}\n")
        
        print(f"实验结果已保存到 {filename}")

    def _build_constraint_violation_operator(self) -> qt.Qobj:
        """Build violation operator $$V = \sum_j ( \sum_i A_{j,i} \hat{n}_i - b_j )^2$$."""
        m_constraints = self.A.shape[0]
        V = 0 * self.n_ops[0]
        for j in range(m_constraints):
            C_j = sum(self.A[j, i] * self.n_ops[i] for i in range(self.num_modes))
            V += (C_j - self.b[j]) ** 2
        return V

    def _get_lindblad_operators(self, error_config: Dict) -> List[qt.Qobj]:
        """Return Lindblad operators L_k for a given error config."""
        error_type = error_config.get('type', '')
        mode = error_config.get('mode', 0)
        rate = error_config.get('rate', 1.0)
        n_th = error_config.get('n_th', 0.5)
        chi = error_config.get('chi', 0.1)
        eta = error_config.get('eta', 0.5)
        imbalance_rate = error_config.get('imbalance_rate', 0.1)
        other_mode = error_config.get('other_mode', 1)  # For cross-mode

        Ls = []

        if error_type == 'photon_loss':
            Ls.append(np.sqrt(rate) * self.a_ops[mode])
        elif error_type == 'photon_gain':
            Ls.append(np.sqrt(rate) * self.ad_ops[mode])
        elif error_type == 'thermal':
            L_down = np.sqrt(rate * (n_th + 1)) * self.a_ops[mode]
            L_up = np.sqrt(rate * n_th) * self.ad_ops[mode]
            Ls.extend([L_down, L_up])
        elif error_type == 'cross_mode_unbalanced':
            # Beam-splitter term + imbalance
            L_bs = np.sqrt(eta) * (self.ad_ops[mode] * self.a_ops[other_mode] + self.a_ops[mode] * self.ad_ops[other_mode])
            L_imbal = np.sqrt(imbalance_rate) * self.a_ops[mode]
            Ls.extend([L_bs, L_imbal])
        elif error_type == 'kerr_loss':
            # Kerr is coherent, added to H; loss as L
            Ls.append(np.sqrt(imbalance_rate) * self.a_ops[mode])  # Reuse imbalance_rate as loss rate
            # Note: chi added to H in simulation call

        return Ls

    def simulate_errors(
        self,
        error_configs: List[Dict],
        tlist: np.ndarray,
        H_evol: Optional[qt.Qobj] = None,
        initial_state: Optional[qt.Qobj] = None,
        plot: bool = True,
        save_path: str = "figs/error_simulation.svg"
    ) -> Dict[str, np.ndarray]:
        """
        Simulate subspace confinement under noisy evolution for multiple error configurations.

        The constraint violation is quantified by the expectation value of the operator
        $$
        \hat{V} = \sum_{j=1}^m \left( \sum_{i=1}^d A_{j,i} \hat{n}_i - b_j \right)^2,
        $$
        where \(\langle \hat{V} \rangle = 0\) indicates perfect confinement in the feasible subspace \(\{ | \mathbf{n} \rangle \mid A \mathbf{n} = \mathbf{b} \}\).

        Args:
            error_configs: List of dicts, each specifying an error, e.g.,
                           [{'type': 'photon_loss', 'mode': 0, 'rate': 1.0},
                            {'type': 'thermal', 'mode': 0, 'rate': 1.0, 'n_th': 0.5}, ...]
            tlist: Time array for evolution.
            H_evol: Coherent Hamiltonian for evolution (default: self.H_M).
            initial_state: Initial state (default: uniform superposition over feasible states).
            plot: Whether to plot violation vs. time.
            save_path: Path to save plot.

        Returns:
            Dict of {error_label: violation(t)} arrays.
        """
        if H_evol is None:
            H_evol = self.H_M
        if initial_state is None:
            # Uniform superposition over feasible states
            psi_list = [qt.tensor(*[qt.fock(self.N, n) for n in ns]) for ns in self.feasible_states]
            initial_state = sum(psi_list) / np.sqrt(len(psi_list))
        rho0 = initial_state * initial_state.dag()

        V = self._build_constraint_violation_operator()  # Violation operator

        violations = {}  # {label: <V>(t)}

        for config in error_configs:
            label = f"{config['type']} (mode {config.get('mode', 0)})"
            Ls = self._get_lindblad_operators(config)

            # For Kerr, add H_kerr to H_evol
            H_total = H_evol
            if config.get('type') == 'kerr_loss':
                chi = config.get('chi', 0.1)
                H_kerr = chi * self.n_ops[config.get('mode', 0)] * (self.n_ops[config.get('mode', 0)] - 1) / 2
                H_total += H_kerr

            # Evolve under Lindblad ME
            result = qt.mesolve(H_total, rho0, tlist, Ls)
            expects = qt.expect(V, result.states)
            violations[label] = expects

        if plot:
            plt.figure(figsize=(10, 6))
            for label, viol_t in violations.items():
                plt.plot(tlist, viol_t, label=label, linewidth=2)
            # Ideal (no noise)
            result_ideal = qt.mesolve(H_evol, rho0, tlist, [])
            viol_ideal = qt.expect(V, result_ideal.states)
            plt.plot(tlist, viol_ideal, 'k--', label='Ideal (No Error)', linewidth=2)
            plt.xlabel('Time $t$')
            plt.ylabel(r'$\langle \hat{V} \rangle$')
            plt.title(r'Constraint Violation $\langle \hat{V} \rangle$ Under Errors')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale for small violations
            plt.tight_layout()
            plt.savefig(save_path)
        

        return violations

# def generate_random_ip_instance(num_variables, num_constraints, max_coeff=5, max_rhs=30):
#     """
#     随机生成整数规划问题实例：A x = b
    
#     Args:
#         num_variables: 变量数量 (d)
#         num_constraints: 约束数量 (m)
#         max_coeff: 系数矩阵A的最大绝对值
#         max_rhs: 右侧向量b的最大绝对值
    
#     Returns:
#         A, b, c: 约束矩阵、右侧向量、目标函数系数
#     """
#     # 生成随机系数矩阵A (整数)
#     A = np.random.randint(1, max_coeff + 1, size=(num_constraints, num_variables))
    
#     # 生成随机可行解x0
#     x0 = np.random.randint(1, 5, size=num_variables)
    
#     # 计算b = A x0，确保存在可行解
#     b = A @ x0
    
#     # 生成随机目标函数系数c
#     c = np.random.randint(1, max_coeff + 1, size=num_variables)
    
#     return A.tolist(), b.tolist(), c.tolist()

def generate_random_ip_instance(num_variables, num_constraints, max_coeff=5, max_rhs=30, N=10, feasible_count=10): 
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
    num_variables = 6      # 变量数量 d
    num_constraints = 3    # 约束数量 m
    set_N = 10
    
    # 支持三种电路类型
    circuit_types = ["multi_beta","beta_gamma",  "multi_beta_oneH"]
    
    # 从命令行参数获取运行参数（用于并行执行）
    import sys
    import os
    import argparse
    run_id = 0
    selected_p = None
    selected_circuit = None
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行QAOA求解器')
    parser.add_argument('run_id', type=int, nargs='?', default=0, help='运行ID')
    parser.add_argument('p', type=int, nargs='?', default=2, help='QAOA层数')
    parser.add_argument('circuit_type', nargs='?', default=circuit_types[0], choices=circuit_types, help='电路类型')
    parser.add_argument('--num_variables', type=int, default=6, help='变量数量')
    parser.add_argument('--num_constraints', type=int, default=3, help='约束数量')
    
    # 如果提供了位置参数，使用它们
    # 这是为了保持向后兼容性
    if len(sys.argv) > 1:
        # 解析位置参数
        args, unknown = parser.parse_known_args()
        run_id = args.run_id
        selected_p = args.p
        selected_circuit = args.circuit_type
        
        # 覆盖默认的变量数和约束数
        num_variables = args.num_variables
        num_constraints = args.num_constraints
        
        # 设置随机种子以确保不同运行ID产生不同结果
        np.random.seed(seed + run_id)
        
        # 重新生成问题实例
        A, b, c = generate_random_ip_instance(num_variables, num_constraints)
        
        # 使用当前工作目录作为保存位置，不再创建子目录
        # 因为run_problem_24.py已经在正确的问题目录中运行
        problem_dir = "."  # 当前目录
        
        print(f"\n运行ID: {run_id}, 层数: {selected_p}, 电路类型: {selected_circuit}")
        print(f"随机生成的整数规划实例：")
        print(f"变量数: {num_variables}, 约束数: {num_constraints}")
        print(f"约束矩阵 A = {A}")
        print(f"右侧向量 b = {b}")
        print(f"目标函数系数 c = {c}")
        print(f"结果保存目录: {os.getcwd()}")
        
        # 保存问题参数到单独文件
        with open(os.path.join(problem_dir, "problem_params.txt"), 'w', encoding='utf-8') as f:
            f.write(f"问题编号: {run_id}\n")
            f.write(f"变量数: {num_variables}\n")
            f.write(f"约束数: {num_constraints}\n")
            f.write(f"约束矩阵 A = {A}\n")
            f.write(f"右侧向量 b = {b}\n")
            f.write(f"目标函数系数 c = {c}\n")
        
        problem_name = f"p{selected_p}_{selected_circuit}"
        print(f"\n========== 测试层数 p={selected_p}, 电路类型: {selected_circuit} ==========\n")
        solver = BosonicQAOAIPSolver(A, b, c, N=set_N, p=selected_p, circuit_type=selected_circuit, 
                                    problem_name=f"{problem_name}")
        ## Hamiltonian of solver
        print("Driver Hamiltonian H_M:")
        print(solver.latex_label_H_M)
        result = solver.optimize()
        
        # 保存结果到当前目录
        plot_path = os.path.join(problem_dir, f"plot_{problem_name}.svg")
        solver.plot_results(save_path=plot_path)
        solver.print_summary()
        
        # 保存结果到文件
        result_file = os.path.join(problem_dir, f"results_{problem_name}.txt")
        solver.save_results_to_file(result_file, A, b, c)
    else:
        # 生成单个问题实例
        A, b, c = generate_random_ip_instance(num_variables, num_constraints)
        
        # 创建问题特定的文件夹（使用默认编号0）
        problem_dir = f"problem_instance_0"
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
        
        # 默认模式：运行所有电路类型
        for circuit_type in circuit_types:
            print(f"\n========== 测试电路类型: {circuit_type} ==========\n")
            for p in range(1, 8):
                problem_name = f"p{p}_{circuit_type}"
                print(f"\n========== 测试层数 p={p} ==========\n")
                solver = BosonicQAOAIPSolver(A, b, c, N=set_N, p=p, circuit_type=circuit_type, 
                                            problem_name=f"{problem_name}")
                ## Hamiltonian of solver
                print("Driver Hamiltonian H_M:")
                print(solver.latex_label_H_M)
                result = solver.optimize()
                
                # 保存结果到问题特定文件夹
                plot_path = os.path.join(problem_dir, f"plot_{problem_name}.svg")
                solver.plot_results(save_path=plot_path)
                solver.print_summary()
                
                # 保存结果到文件
                result_file = os.path.join(problem_dir, f"results_{problem_name}.txt")
                solver.save_results_to_file(result_file, A, b, c)

    # Example error simulation
    # error_configs = [
    #     {'type': 'photon_loss', 'mode': 0, 'rate': 1.0},
    #     {'type': 'photon_gain', 'mode': 0, 'rate': 1.0},
    #     {'type': 'thermal', 'mode': 0, 'rate': 1.0, 'n_th': 0.5},
    #     {'type': 'cross_mode_unbalanced', 'mode': 0, 'other_mode': 1, 'eta': 0.5, 'imbalance_rate': 0.1},
    #     {'type': 'kerr_loss', 'mode': 0, 'chi': 0.1, 'imbalance_rate': 0.05}
    # ]
    # tlist = np.linspace(0, 0.1, 50)
    # violations = solver.simulate_errors(error_configs, tlist, H_evol=solver.H_M)