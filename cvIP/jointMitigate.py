import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp
from typing import List, Tuple, Optional, Union, Dict
from scipy.linalg import eigh

class BosonicQAOAIPSolver:
    def __init__(self, A, b, c, N=3, p=1, g=1.0, maxiter=50, seed=42, num_modes=None, circuit_type="beta_gamma"):
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.c = np.array(c, dtype=float)
        self.N = N
        self.p = p
        self.g = g
        self.maxiter = maxiter
        self.seed = seed
        self.num_modes = num_modes or self.c.shape[0] or self.A.shape[1]
        np.random.seed(seed)
        self.null_space_basis = self._compute_integer_nullspace()
        self.a_ops, self.ad_ops, self.n_ops = self._build_operators()
        self.H_C = self._build_cost_hamiltonian()
        self.H_ds = self._build_seperate_driver_hamiltonian()
        self.H_M = sum(self.H_ds)
        self.circuit_type = circuit_type
        if circuit_type == "beta_gamma":
            self.qaoa_circuit = self.qaoa_circuit_beta_gamma
            self.paramlength = 2
        self.feasible_states = self._find_feasible_states()
        self.initial_state = min(self.feasible_states, key=lambda ns: sum(self.c * ns))
        self.tracked_states = sorted(self.feasible_states, key=lambda ns: sum(self.c * ns), reverse=True)[:6] + [self.initial_state]
        self.tracked_labels = [f"|{','.join(map(str, ns))}⟩ (obj={sum(self.c * ns):.1f})" for ns in self.tracked_states]

    def _compute_integer_nullspace(self):
        A_sym = sp.Matrix(self.A)
        ns_rational = A_sym.nullspace()
        if not ns_rational:
            return np.array([[1, -1]]).T  # Dummy for demo
        int_vecs = []
        for vec in ns_rational:
            int_vec = np.array(vec).flatten().astype(int)
            gcd = np.gcd.reduce(int_vec)
            if gcd != 0:
                int_vec //= gcd
            int_vecs.append(int_vec)
        return np.array(int_vecs)

    def _build_operators(self):
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

    def _build_cost_hamiltonian(self):
        return sum(-self.c[i] * self.n_ops[i] for i in range(self.num_modes))

    def _build_seperate_driver_hamiltonian(self):
        Hds = []
        I_multi = self._build_identity()
        for u in self.null_space_basis:
            O_u = I_multi.copy()
            for i in range(self.num_modes):
                if u[i] > 0:
                    O_u = O_u * (self.ad_ops[i] ** u[i])
                elif u[i] < 0:
                    O_u = O_u * (self.a_ops[i] ** abs(u[i]))
            H_M = self.g * (O_u + O_u.dag())
            Hds.append(H_M)
        return Hds

    def _build_identity(self):
        return qt.tensor([qt.qeye(self.N) for _ in range(self.num_modes)])

    def _find_feasible_states(self):
        feasible = []
        for ns_tuple in product(range(self.N), repeat=self.num_modes):
            ns = np.array(ns_tuple)
            if np.allclose(self.A @ ns, self.b):
                feasible.append(tuple(ns))
        return feasible

    def qaoa_circuit_beta_gamma(self, params, initial_state):
        state = initial_state.copy()
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            state = (-1j * self.H_C * gamma).expm() * state
            state = (-1j * self.H_M * beta).expm() * state
        return state

    def optimize(self, initial_state=None):
        if initial_state is None:
            initial_state = qt.tensor(*[qt.fock(self.N, n) for n in self.initial_state])
        init_params = np.random.uniform(0, np.pi, self.paramlength * self.p)
        result = minimize(lambda params: qt.expect(self.H_C, self.qaoa_circuit(params, initial_state)), init_params, method="COBYLA", options={"maxiter": self.maxiter})
        self.optimal_params = result.x
        self.final_state = self.qaoa_circuit(self.optimal_params, initial_state)
        self.final_cost = result.fun
        self.final_obj = -self.final_cost
        return {"params": self.optimal_params, "obj": self.final_obj, "state": self.final_state}

    def _build_constraint_violation_operator(self):
        m = self.A.shape[0]
        V = 0 * self.n_ops[0]
        for j in range(m):
            C_j = sum(self.A[j, i] * self.n_ops[i] for i in range(self.num_modes))
            V += (C_j - self.b[j]) ** 2
        return V

    def _get_lindblad_operators(self, configs):
        Ls = []
        for config in configs:
            error_type = config.get('type', '')
            mode = config.get('mode', 0)
            rate = config.get('rate', 1.0)
            n_th = config.get('n_th', 0.5)
            if error_type == 'photon_loss':
                Ls.append(np.sqrt(rate) * self.a_ops[mode])
            elif error_type == 'photon_gain':
                Ls.append(np.sqrt(rate) * self.ad_ops[mode])
            elif error_type == 'thermal':
                L_down = np.sqrt(rate * (n_th + 1)) * self.a_ops[mode]
                L_up = np.sqrt(rate * n_th) * self.ad_ops[mode]
                Ls.extend([L_down, L_up])
        return Ls

    def mitigate_gse_matrices(self, rho_noisy: qt.Qobj, O_dict: Dict[str, qt.Qobj], K: int = 2, A: Optional[qt.Qobj] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute overlap matrix S and observable matrices O_mats for GSE power subspace.
        O_dict: {'HC': H_C, 'V': V}.
        """
        if A is None:
            A = self._build_identity()
        sigmas = []
        rho_current = self._build_identity()
        for i in range(K + 1):
            sigmas.append(rho_current)
            rho_current = rho_current * rho_noisy
        d_s = len(sigmas)
        S = np.zeros((d_s, d_s), dtype=complex)
        O_mats = {name: np.zeros((d_s, d_s), dtype=complex) for name in O_dict}
        for i in range(d_s):
            sigma_i_dag = sigmas[i].dag()
            for j in range(d_s):
                temp = sigma_i_dag * A * sigmas[j]
                S[i, j] = temp.tr().real
                for name, O in O_dict.items():
                    O_mats[name][i, j] = (temp * O).tr()
        return S, O_mats

    def mitigate_gse_joint(self, rho_noisy: qt.Qobj, HC: qt.Qobj, V: qt.Qobj, lambda_penalty: float = 10.0, K: int = 2, A: Optional[qt.Qobj] = None) -> Tuple[float, float]:
        """
        Joint GSE: Minimize <H_C + λ V>, return <H_C>_EM, <V>_EM.
        """
        O_dict = {'HC': HC, 'V': V}
        S, O_mats = self.mitigate_gse_matrices(rho_noisy, O_dict, K=K, A=A)
        S += 1e-10 * np.eye(S.shape[0])  # Positive-definite regularization
        H_target_mat = O_mats['HC'] + lambda_penalty * O_mats['V']
        evals, evecs = eigh(H_target_mat, S)
        idx_min = np.argmin(evals.real)
        alpha = evecs[:, idx_min]
        norm = np.sqrt(np.real(alpha.conj().T @ S @ alpha))
        if norm > 1e-10:
            alpha /= norm
        hc_gse = np.real(alpha.conj().T @ O_mats['HC'] @ alpha)
        v_gse = np.real(alpha.conj().T @ O_mats['V'] @ alpha)
        return hc_gse, v_gse

    def mitigate_gse_vfirst(self, rho_noisy: qt.Qobj, HC: qt.Qobj, V: qt.Qobj, K: int = 2, A: Optional[qt.Qobj] = None) -> Tuple[float, float]:
        """
        Violation-first GSE: Minimize <V>, then post-hoc <H_C>_EM.
        """
        O_dict = {'HC': HC, 'V': V}
        S, O_mats = self.mitigate_gse_matrices(rho_noisy, O_dict, K=K, A=A)
        S += 1e-10 * np.eye(S.shape[0])
        evals, evecs = eigh(O_mats['V'], S)
        idx_min = np.argmin(evals.real)
        alpha = evecs[:, idx_min]
        norm = np.sqrt(np.real(alpha.conj().T @ S @ alpha))
        if norm > 1e-10:
            alpha /= norm
        hc_gse = np.real(alpha.conj().T @ O_mats['HC'] @ alpha)
        v_gse = np.real(alpha.conj().T @ O_mats['V'] @ alpha)
        return hc_gse, v_gse

    # [Full class definition as provided, with additions below]

    def mitigate_dual_gse_matrices(self, rho_noisy: qt.Qobj, O_dict: Dict[str, qt.Qobj], K: int = 1, A: Optional[qt.Qobj] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Dual-GSE power subspace matrices with DSP symmetrization (K=1)."""
        if A is None:
            A = self._build_identity()
        bar_rho = rho_noisy  # Approximation for simulation
        sigma0 = A  # I
        sigma1 = rho_noisy
        rho2_sym = (bar_rho * rho_noisy + rho_noisy * bar_rho) / 2
        rho2_sym = rho2_sym / rho2_sym.tr()  # Normalize
        sigmas = [sigma0, sigma1, rho2_sym]
        d_s = len(sigmas)
        S = np.zeros((d_s, d_s), dtype=complex)
        O_mats = {name: np.zeros((d_s, d_s), dtype=complex) for name in O_dict}
        for i in range(d_s):
            sigma_i_dag = sigmas[i].dag()
            for j in range(d_s):
                temp = sigma_i_dag * A * sigmas[j]
                S[i, j] = temp.tr().real
                for name, O in O_dict.items():
                    O_mats[name][i, j] = (temp * O).tr()
        return S, O_mats

    def mitigate_dual_gse_joint(self, rho_noisy: qt.Qobj, HC: qt.Qobj, V: qt.Qobj, lambda_penalty: float = 10.0, K: int = 1, A: Optional[qt.Qobj] = None) -> Tuple[float, float]:
        """Joint Dual-GSE with DSP."""
        O_dict = {'HC': HC, 'V': V}
        S, O_mats = self.mitigate_dual_gse_matrices(rho_noisy, O_dict, K=K, A=A)
        S += 1e-10 * np.eye(S.shape[0])
        H_target_mat = O_mats['HC'] + lambda_penalty * O_mats['V']
        evals, evecs = eigh(H_target_mat, S)
        idx_min = np.argmin(evals.real)
        alpha = evecs[:, idx_min]
        norm = np.sqrt(np.real(alpha.conj().T @ S @ alpha))
        if norm > 1e-10:
            alpha /= norm
        hc_gse = np.real(alpha.conj().T @ O_mats['HC'] @ alpha)
        v_gse = np.real(alpha.conj().T @ O_mats['V'] @ alpha)
        return hc_gse, v_gse

    def mitigate_dual_gse_vfirst(self, rho_noisy: qt.Qobj, HC: qt.Qobj, V: qt.Qobj, K: int = 1, A: Optional[qt.Qobj] = None) -> Tuple[float, float]:
        """Violation-first Dual-GSE with DSP."""
        O_dict = {'HC': HC, 'V': V}
        S, O_mats = self.mitigate_dual_gse_matrices(rho_noisy, O_dict, K=K, A=A)
        S += 1e-10 * np.eye(S.shape[0])
        evals, evecs = eigh(O_mats['V'], S)
        idx_min = np.argmin(evals.real)
        alpha = evecs[:, idx_min]
        norm = np.sqrt(np.real(alpha.conj().T @ S @ alpha))
        if norm > 1e-10:
            alpha /= norm
        hc_gse = np.real(alpha.conj().T @ O_mats['HC'] @ alpha)
        v_gse = np.real(alpha.conj().T @ O_mats['V'] @ alpha)
        return hc_gse, v_gse

# Modified simulate_errors to include Dual-GSE (add to the loop):
# dual_HC[idx], dual_V[idx] = self.mitigate_dual_gse_joint(rho_t, HC, V_op, lambda_penalty=lambda_penalty, K=1)
# Then compute mse_dual similarly.

    def _get_lindblad_operators(self, configs: List[Dict]) -> List[qt.Qobj]:
        """
        Aggregate Lindblad operators L_k from multiple configs for concurrent noise.
        """
        Ls = []
        for config in configs:
            typ = config.get('type', '')
            mode = config.get('mode', 0)
            rate = config.get('rate', 1.0)
            n_th = config.get('n_th', 0.5)
            chi = config.get('chi', 0.1)
            eta = config.get('eta', 0.5)
            imbalance_rate = config.get('imbalance_rate', 0.1)
            other_mode = config.get('other_mode', 1)
            if typ == 'photon_loss':
                Ls.append(np.sqrt(rate) * self.a_ops[mode])
            elif typ == 'photon_gain':
                Ls.append(np.sqrt(rate) * self.ad_ops[mode])
            elif typ == 'thermal':
                L_down = np.sqrt(rate * (n_th + 1)) * self.a_ops[mode]
                L_up = np.sqrt(rate * n_th) * self.ad_ops[mode]
                Ls.extend([L_down, L_up])
            elif typ == 'cross_mode_unbalanced':
                L_bs = np.sqrt(eta) * (self.ad_ops[mode] * self.a_ops[other_mode] + self.a_ops[mode] * self.ad_ops[other_mode])
                L_imbal = np.sqrt(imbalance_rate) * self.a_ops[mode]
                Ls.extend([L_bs, L_imbal])
            elif typ == 'kerr_loss':
                Ls.append(np.sqrt(imbalance_rate) * self.a_ops[mode])
                # Kerr added to H_total in caller
        return Ls

    def simulate_errors(
        self,
        error_configs: List[Dict],
        tlist: np.ndarray,
        H_evol: Optional[qt.Qobj] = None,
        initial_state: Optional[qt.Qobj] = None,
        gse_K: int = 2,
        lambda_penalty: float = 10.0,
        plot: bool = True,
        save_path: str = "plots/figs/gse_comparison.svg"
    ) -> Dict[str, np.ndarray]:
        """
        Simulate noisy evolution with joint vs. violation-first GSE, supporting multi-error configs.

        Aggregates L_k for concurrent noise; computes raw/mitigated <H_C>(t), <V>(t).

        Returns dict with 'raw_HC', 'raw_V', 'joint_HC', 'joint_V', 'vfirst_HC', 'vfirst_V',
        'mse_joint', 'mse_vfirst' (combined MSE = 0.5 (||ΔH_C||^2 + ||ΔV||^2)).
        """
        if H_evol is None:
            H_evol = self.H_M
        if initial_state is None:
            # psi_list = [qt.tensor(*[qt.fock(self.N, n) for n in ns]) for ns in self.feasible_states]
            initial_state =  qt.tensor(*[qt.fock(self.N, n) for n in self.initial_state])
        rho0 = initial_state * initial_state.dag()

        V_op = self._build_constraint_violation_operator()
        HC = self.H_C
        Ls = self._get_lindblad_operators(error_configs)  # Multi-error aggregation
        ideal_result = qt.mesolve(H_evol, rho0, tlist, [])
        ideal_HC = qt.expect(HC, ideal_result.states)
        ideal_V = qt.expect(V_op, ideal_result.states)

        # Handle Kerr in multi-config (add to H if present)
        H_total = H_evol
        for config in error_configs:
            if config.get('type') == 'kerr_loss':
                mode = config.get('mode', 0)
                chi = config.get('chi', 0.1)
                H_kerr = chi * self.n_ops[mode] * (self.n_ops[mode] - 1) / 2
                H_total += H_kerr
                break  # Assume one Kerr per call

        result = qt.mesolve(H_total, rho0, tlist, Ls)
        raw_HC = qt.expect(HC, result.states)
        raw_V = qt.expect(V_op, result.states)

        joint_HC = np.zeros_like(raw_HC)
        joint_V = np.zeros_like(raw_V)
        vfirst_HC = np.zeros_like(raw_HC)
        vfirst_V = np.zeros_like(raw_V)
        joint_dual_HC = np.zeros_like(raw_HC)
        joint_dual_V = np.zeros_like(raw_V)
        vfirst_dual_HC = np.zeros_like(raw_HC)
        vfirst_dual_V = np.zeros_like(raw_V)

        for idx, rho_t in enumerate(result.states):
            joint_HC[idx], joint_V[idx] = self.mitigate_gse_joint(rho_t, HC, V_op, lambda_penalty=lambda_penalty, K=gse_K)
            vfirst_HC[idx], vfirst_V[idx] = self.mitigate_gse_vfirst(rho_t, HC, V_op, K=gse_K)
            # Dual-GSE (K=1)
            joint_dual_HC[idx], joint_dual_V[idx] = self.mitigate_dual_gse_joint(rho_t, HC, V_op, lambda_penalty=lambda_penalty, K=3)
            vfirst_dual_HC[idx], vfirst_dual_V[idx] = self.mitigate_dual_gse_vfirst(rho_t, HC, V_op, K=3)
        # Ideal baselines (from t=0, no noise)
        mse_joint = 0.5 * (np.mean((joint_HC - ideal_HC)**2) + np.mean(joint_V**2))
        mse_vfirst = 0.5 * (np.mean((vfirst_HC - ideal_HC)**2) + np.mean(vfirst_V**2))

        if plot:
            width_pt = 240
            inches_per_pt = 1 / 72.27
            fig_width = width_pt * inches_per_pt

            # 计算子图尺寸（16:9比例）
            subplot_width = fig_width / 2  # 4列
            subplot_height = subplot_width *0.8
            fig_height = subplot_height  # 3行
            fontsize = 7
            plt.rcParams.update({
                "font.family": "Arial",
                "font.size": fontsize,
                "figure.figsize": (fig_width, fig_height),
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
                ## tick label padding
                "xtick.major.pad": 0.3,
                "ytick.major.pad": 0.3,
                "legend.fontsize": fontsize,
                "axes.titlesize": fontsize,
                "lines.markersize": 2.5,
                "lines.linewidth": 0.7,
                "lines.markeredgewidth": 0,
                "grid.linewidth": 0.2,
                "grid.alpha": 0.5,
                "grid.color": "gray",
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "hatch.color": "black",
                "hatch.linewidth": 0.5,
            })
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(tlist, ideal_HC, 'k-.', label='noise-free')
            ax1.plot(tlist, raw_HC, '-',color = "#440154", label='with noise')
            # ax1.plot(tlist, joint_HC, 'b--', label='Joint GSE')
            # ax1.plot(tlist, vfirst_HC, 'r-.', label='V-First GSE')
            # ax1.plot(tlist, joint_dual_HC, 'c--', label='Joint Dual-GSE')
            ax1.plot(tlist, vfirst_dual_HC, '-.', color = "#1F77B4",  label='with Drift Manager')
            ax1.set_xlabel('Time $t$'); ax1.set_ylabel(r'$\langle \hat{H}_C \rangle$'); ax1.legend(); ax1.grid(alpha=0.3)
            
            ax2.plot(tlist, ideal_V, '-.', label='noise-free')
            ax2.plot(tlist, raw_V, '-',color = "#440154", label='with noise')
            # ax2.plot(tlist, joint_V, 'b--', label='Joint GSE')
            # ax2.plot(tlist, vfirst_V, 'r-.', label='V-First GSE')
            # ax2.plot(tlist, joint_dual_V, 'c--', label='Joint Dual-GSE')
            ax2.plot(tlist, vfirst_dual_V, '-.', color = "#1F77B4", label='with Drift Manager')
            ax2.set_xlabel('Time $t$'); ax2.set_ylabel(r'$\langle \hat{V} \rangle$'); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_yscale('log')

            # plt.suptitle(f'GSE Comparison (K={gse_K}, λ={lambda_penalty})')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()

        return {
            'raw_HC': raw_HC, 'raw_V': raw_V,
            'joint_HC': joint_HC, 'joint_V': joint_V,
            'vfirst_HC': vfirst_HC, 'vfirst_V': vfirst_V,
            'mse_joint': mse_joint, 'mse_vfirst': mse_vfirst
        }
# Test with asymmetric c = [2, -1]
A_test = [[1, 1,-1]]
b_test = [1]
c_test = [2.0, -1.0, 1]
solver_test = BosonicQAOAIPSolver(A_test, b_test, c_test)
error_configs_test = [
        {'type': 'photon_loss', 'mode': 0, 'rate': 0.1},      # Single-mode loss
        {'type': 'photon_gain', 'mode': 1, 'rate': 0.1},       # Single-mode gain
        {'type': 'photon_loss', 'mode': 2, 'rate': 0.1},      # Single-mode loss
        {'type': 'thermal', 'mode': 0, 'rate': 0.1, 'n_th': 0.1},  # Thermal noise
        {'type': 'cross_mode_unbalanced', 'mode': 0, 'other_mode': 1, 'eta': 0.3, 'imbalance_rate': 0.2},
        {'type': 'kerr_loss', 'mode': 0, 'chi': 0.1, 'imbalance_rate': 0.1}  # Kerr + loss
    ]
tlist_test = np.linspace(0, 2.5, 100)
results_test = solver_test.simulate_errors(error_configs_test, tlist_test, initial_state=None,gse_K=6, lambda_penalty=2)
print("Raw HC:", results_test['raw_HC'])
print("Raw V:", results_test['raw_V'])
print("Joint HC:", results_test['joint_HC'])
print("Joint V:", results_test['joint_V'])
print("VFirst HC:", results_test['vfirst_HC'])
print("VFirst V:", results_test['vfirst_V'])
# Example: Asymmetric objective, multi-error
# A = [[1, 1]]; b = [1]; c = [2.0, -1.0]  # <H_C> = -2 n1 + n2, ideal -0.5
# solver = BosonicQAOAIPSolver(A, b, c, N=5, p=1)  # N=5 for distinction
# configs = [
#     {'type': 'photon_loss', 'mode': 0, 'rate': 0.4},  # Loss on high-c mode
#     {'type': 'thermal', 'mode': 1, 'rate': 0.3, 'n_th': 0.2}  # Thermal on low-c mode
# ]
# tlist = np.linspace(0, 0.3, 10)
# results = solver.simulate_errors(configs, tlist, initial_state=None, gse_K=4, lambda_penalty=5.0)
# print(f"Joint MSE: {results['mse_joint']:.6f}, V-First MSE: {results['mse_vfirst']:.6f}")