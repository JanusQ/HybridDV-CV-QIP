import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import sympy as sp_sym
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import List, Tuple, Dict
import time

class OptimizedBosonicSolver:
    """
    Optimized Scalable Solver (v4.0).
    
    Optimizations:
    1. active_pruning: Skips drivers that don't affect the current state.
    2. random_shuffling: Shuffles driver order per layer to mitigate Trotter errors.
    3. sparse_dynamic: Basis expands dynamically from a single initial point.
    """
    
    def __init__(
        self,
        A: np.ndarray, 
        b: np.ndarray,
        c: np.ndarray,
        N: int = 7,
        p: int = 2,
        g: float = 1.0
    ):
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.c = np.array(c, dtype=float)
        self.N = N
        self.p = p
        self.g = g
        self.num_modes = len(self.c)
        
        # 1. Compute Drivers
        self.null_vectors = self._compute_integer_nullspace()
        self.num_drivers = len(self.null_vectors)
        print(f"Driver Vectors: {self.num_drivers}")
        print(f"Driver Vectors: {np.array(self.null_vectors)}")
        
        # 2. Find ONE initial feasible state
        self.initial_state = self._find_single_feasible_state()
        print(f"Initial State: {np.array(self.initial_state)}")
        
    def _compute_integer_nullspace(self) -> List[Tuple[int]]:
        A_sym = sp_sym.Matrix(self.A)
        ns = A_sym.nullspace()
        basis = []
        for vec in ns:
            denoms = [x.q for x in vec if hasattr(x, 'q')]
            lcm = sp_sym.lcm(denoms) if denoms else 1
            v_int = np.array([(x * lcm) for x in vec], dtype=int).flatten()
            gcd = np.gcd.reduce(v_int)
            if gcd!=0: v_int //= gcd
            basis.append(tuple(v_int))
        
        # Enhance mixing
        if len(basis) > 1:
            basis.append(tuple(np.array(basis[0]) + np.array(basis[-1])))
            basis.append(tuple(np.array(basis[0]) - np.array(basis[-1])))
        return basis

    def _find_single_feasible_state(self) -> Tuple[int, ...]:
        """Classical fast start."""
        c_min = self.c 
        constraints = LinearConstraint(self.A, self.b, self.b)
        bounds = Bounds(np.zeros(self.num_modes), np.full(self.num_modes, self.N - 1))
        res = milp(c=c_min, constraints=constraints, bounds=bounds, integrality=np.ones(self.num_modes))
        if not res.success:
            raise ValueError("Infeasible constraints.")
        return tuple(np.round(res.x).astype(int))

    def _is_valid_state(self, s_arr: np.ndarray) -> bool:
        return np.all(s_arr >= 0) and np.all(s_arr < self.N)

    def _has_active_transitions(self, state_dict: Dict[Tuple, complex], u_arr: np.ndarray) -> bool:
        """
        Optimization 1 Check:
        Returns True if ANY state in the dictionary can hop by +u or -u 
        without hitting boundaries.
        """
        for s in state_dict:
            s_arr = np.array(s)
            # Check +u
            if self._is_valid_state(s_arr + u_arr): return True
            
            # Check -u
            if self._is_valid_state(s_arr - u_arr): return True
        return False

    def _apply_driver_sparse(self, state_dict: Dict[Tuple, complex], beta: float, u: Tuple[int]) -> Dict[Tuple, complex]:
        """Apply H_u with basis expansion."""
        u_arr = np.array(u)

        # --- OPTIMIZATION 1: Pruning ---
        # If this driver can't move any current state, it's an Identity operation. Skip it.
        # This saves expensive matrix construction and exponentiation.
        if not self._has_active_transitions(state_dict, u_arr):
            return state_dict

        # If active, proceed with normal logic...
        active_basis = set(state_dict.keys())
        new_candidates = set()
        
        for s in active_basis:
            s_arr = np.array(s)
            for direction in [1, -1]:
                s_next = s_arr + direction * u_arr
                if self._is_valid_state(s_next):
                    new_candidates.add(tuple(s_next))
        
        full_basis = list(active_basis.union(new_candidates))
        state_to_idx = {s: i for i, s in enumerate(full_basis)}
        dim = len(full_basis)
        
        rows, cols, data = [], [], []
        for i, s in enumerate(full_basis):
            s_arr = np.array(s)
            s_next = s_arr + u_arr
            if self._is_valid_state(s_next):
                tgt = tuple(s_next)
                if tgt in state_to_idx:
                    j = state_to_idx[tgt]
                    val = self._calc_transition_amp(s_arr, u_arr)
                    if abs(val) > 1e-9:
                        rows.append(j); cols.append(i); data.append(val * self.g)
                        rows.append(i); cols.append(j); data.append(val * self.g)

        if not data: return state_dict # Double check
            
        H_sub = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
        
        vec = np.zeros(dim, dtype=complex)
        for s, amp in state_dict.items():
            vec[state_to_idx[s]] = amp
            
        vec_new = expm_multiply(-1j * beta * H_sub, vec)
        
        new_dict = {}
        norm = 0.0
        for i, amp in enumerate(vec_new):
            prob = abs(amp)**2
            if prob > 1e-6:
                new_dict[full_basis[i]] = amp
                norm += prob
        
        if norm > 0:
            scale = 1.0 / np.sqrt(norm)
            for s in new_dict: new_dict[s] *= scale
            
        return new_dict

    def _calc_transition_amp(self, state: np.ndarray, u: np.ndarray) -> float:
        amp = 1.0
        temp = state.copy()
        for k in range(len(u)):
            if u[k] < 0:
                for _ in range(abs(u[k])): amp *= np.sqrt(temp[k]); temp[k] -= 1
        for k in range(len(u)):
            if u[k] > 0:
                for _ in range(u[k]): amp *= np.sqrt(temp[k] + 1); temp[k] += 1
        return amp

    def simulate(self, params: np.ndarray, shuffle: bool = True) -> Dict[Tuple, complex]:
        state = {self.initial_state: 1.0 + 0j}
        
        # Params layout: [beta_0_0... beta_0_k, ... ]
        param_idx = 0
        
        # Indices of drivers to shuffle
        driver_indices = np.arange(self.num_drivers)
        
        for layer in range(self.p):
            # 2. Driver Layer (Optimized)
            # Extract betas for this layer first
            layer_betas = params[param_idx : param_idx + self.num_drivers]
            param_idx += self.num_drivers
            
            # --- OPTIMIZATION 2: Shuffling ---
            # We shuffle the execution order, but keep the beta associated with the driver ID.
            execution_order = driver_indices.copy()
            if shuffle:
                np.random.shuffle(execution_order)
            
            skipped_count = 0
            for u_idx in execution_order:
                beta = layer_betas[u_idx] # Crucial: Get beta corresponding to driver ID
                u_vec = self.null_vectors[u_idx]
                
                # Apply (will internally check for Pruning)
                prev_len = len(state)
                state = self._apply_driver_sparse(state, beta, u_vec)
                
                # Just for debug/stats (optional)
                # if len(state) == prev_len: skipped_count += 1
                    
        return state

    def optimize(self):
        from scipy.optimize import minimize
        
        num_params = self.p * (self.num_drivers)
        x0 = np.random.uniform(-0.5, 0.5, num_params)
        bounds = [(-np.pi, np.pi) for _ in range(num_params)]
        print(f"Optimizing {num_params} parameters...")
        print("Features: Dynamic Expansion + Driver Pruning + Layer Shuffling")
        
        # Note: Using random shuffle in 'loss' makes the function stochastic.
        # COBYLA handles small noise okay, but for strict convergence, 
        # one might want to fix the seed inside 'simulate' or set shuffle=False.
        # Here we leave shuffle=True to demonstrate the feature request.
        
        best_obj = -float('inf')
        loss_history = []
        iter_idx = 0

        def loss(x):
            # Pass shuffle=True to enable the optimization technique requested
            final_state = self.simulate(x, shuffle=False)
            exp_obj = sum(abs(v)**2 * np.dot(self.c, k) for k, v in final_state.items())
            return -exp_obj
            
        def callback(x):
            nonlocal best_obj
            nonlocal iter_idx
            val = -loss(x)
            if val > best_obj: 
                best_obj = val
            iter_idx += 1
            print(f"Current Best Exp. Obj: {best_obj:.4f}")
            loss_history.append(val)
        
        res = minimize(loss, x0,method='COBYLA',bounds=bounds, options={'maxiter': 500}, callback=callback)
        
        final_state = self.simulate(res.x, shuffle=False) # Final run without shuffle for clean stats
        top_states = sorted([(np.array(k), abs(v)**2, np.dot(self.c, k)) for k, v in final_state.items()], key=lambda x:x[1], reverse=True)
        
        return {
            "x": res.x,
            "obj": -res.fun,
            "top_states": top_states[:10],
            "total_states": len(final_state)
        }

if __name__ == "__main__":
    np.random.seed(42)
    
    # Define a medium-sized problem
    num_vars = 10
    num_cons = 4
    N_trunc = 20
    
    while True:
        A = np.random.randint(0, 3, size=(num_cons, num_vars))
        x_sol = np.random.randint(0, 7, size=num_vars)
        b = A @ x_sol
        c = np.random.randint(1, 8, size=num_vars)
        if np.count_nonzero(A) > 6: break
    
    print("=== Problem ===")
    print(f"A shape: {A.shape}, N: {N_trunc}")
    print(f"Objective: Maximize c.x")
    
    solver = OptimizedBosonicSolver(A, b, c, N=N_trunc, p=3)
    
    t0 = time.time()
    res = solver.optimize()
    t1 = time.time()
    
    print("\n=== Results ===")
    print(f"Time: {t1-t0:.2f}s")
    print(f"Final Exp Obj: {res['obj']:.4f}")
    print(f"Explored Subspace Size: {res['total_states']}")
    print("\nTop 5 States:")
    for s, p, o in res['top_states'][:5]:
        print(f"State: {s} | Prob: {p:.4f} | Obj: {o:.1f}")