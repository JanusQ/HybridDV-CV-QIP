import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os

# Utility to ensure output directory
def ensure_dir(directory: str) -> None:
    """Ensure the output directory exists."""
    os.makedirs(directory, exist_ok=True)

class BosonicDriverSimulator:
    """
    Simulator for the driver Hamiltonian evolution in multi-mode bosonic systems,
    enforcing subspace confinement for linear constraints sum c_i n_i = gamma.
    
    Attributes:
        N (int): Fock truncation per mode.
        num_modes (int): Number of bosonic modes M.
        constraint_coeffs (List[float]): Coefficients c = [c_1, ..., c_M].
        target_value (float): Constraint value gamma.
        null_space_basis (List[np.ndarray]): Integer basis vectors u in ker(c^T).
        g (float): Uniform coupling strength.
        times (np.ndarray): Evolution times t.
        a_ops (List[qt.Qobj]): Annihilation operators a_i.
        ad_ops (List[qt.Qobj]): Creation operators a_i^dagger.
        H_d (qt.Qobj): Driver Hamiltonian.
    """
    
    def __init__(
        self,
        N: int,
        constraint_coeffs: List[float],
        target_value: float,
        null_space_basis: List[np.ndarray],
        g: float = 1.0,
        times: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        self.N = N
        self.num_modes = len(constraint_coeffs)
        self.constraint_coeffs = np.array(constraint_coeffs)
        self.target_value = target_value
        self.null_space_basis = [np.array(u) for u in null_space_basis]
        self.g = g
        self.times = times if times is not None else np.linspace(0, 100, 200)
        np.random.seed(seed)
        
        # Validate null space
        for u in self.null_space_basis:
            if not np.isclose(np.dot(self.constraint_coeffs, u), 0):
                raise ValueError(f"Vector {u} not in null space of {self.constraint_coeffs}.")
        
        # Build operators
        self._build_operators()
        self.H_d = self._build_hamiltonian()
        print(f"Driver Hamiltonian H_d constructed (norm: {self.H_d.norm():.2f}).")
        print(f"Null space basis: {self.null_space_basis}")
        print(f"Constraint: {' + '.join(f'{c}*n_{i+1}' for i, c in enumerate(self.constraint_coeffs))} = {self.target_value}")
    
    def _build_operators(self) -> None:
        """Construct tensor-product annihilation and creation operators."""
        self.a_ops = []
        self.ad_ops = []
        for i in range(self.num_modes):
            ops = [qt.qeye(self.N)] * self.num_modes
            ops[i] = qt.destroy(self.N)
            self.a_ops.append(qt.tensor(ops))
            self.ad_ops.append(self.a_ops[-1].dag())
    
    def _build_operator_u(self, u: np.ndarray) -> qt.Qobj:
        """Build O_u = prod_i (a_i^dagger)^{u_i} if u_i > 0 else a_i^{|u_i|} else id."""
        O_u = qt.qeye(1)  # Scalar identity
        for i in range(self.num_modes):
            if u[i] > 0:
                O_u = O_u * (self.ad_ops[i] ** u[i])
            elif u[i] < 0:
                O_u = O_u * (self.a_ops[i] ** abs(u[i]))
        return O_u
    
    def _build_hamiltonian(self) -> qt.Qobj:
        """Construct H_d = sum_u g (O_u + O_u^dagger)."""
        H_d = 0 * self.a_ops[0]  # Zero operator
        for u in self.null_space_basis:
            O_u = self._build_operator_u(u)
            H_u = self.g * (O_u + O_u.dag())
            H_d += H_u
        return H_d
    
    def create_initial_state(
        self,
        fock_vecs: List[List[int]],
        superposition: bool = True
    ) -> qt.Qobj:
        """Create initial state |psi_0> = sum_k |n^{(k)}> / sqrt(K) for superposition, or first vec."""
        states = [qt.tensor(*[qt.fock(self.N, n) for n in vec]) for vec in fock_vecs]
        if superposition:
            psi_0 = sum(states).unit()
        else:
            psi_0 = states[0]
        # Verify constraint
        for vec in fock_vecs:
            if not np.isclose(sum(self.constraint_coeffs * np.array(vec)), self.target_value):
                raise ValueError(f"State {vec} violates constraint.")
        return psi_0
    
    def evolve(self, initial_state: qt.Qobj, c_ops: List[qt.Qobj] = None) -> qt.Result:
        """Evolve |psi(t)> under H_d using mesolve (c_ops for optional dissipation)."""
        if c_ops is None:
            c_ops = []
        return qt.mesolve(self.H_d, initial_state, self.times, c_ops, [])
    
    def compute_fock_probs(self, rho: qt.Qobj) -> np.ndarray:
        """Compute P(n_1, n_2, ..., n_M) = |<n|rho|n>|^2 efficiently via precomputed basis."""
        # Precompute all Fock basis states once (for reuse in batch)
        basis_states = np.empty((self.N ** self.num_modes,), dtype=object)
        idx = 0
        for ns in product(range(self.N), repeat=self.num_modes):
            basis_states[idx] = qt.tensor(*[qt.fock(self.N, n) for n in ns])
            idx += 1
        probs = np.array([abs(b.overlap(rho)) ** 2 for b in basis_states])
        return probs.reshape([self.N] * self.num_modes)
    
    def compute_satisfaction_rate(self, probs: np.ndarray) -> float:
        """Compute fraction of probability in valid subspace S_c."""
        valid_mask = np.zeros_like(probs, dtype=bool)
        for ns in product(range(self.N), repeat=self.num_modes):
            if np.isclose(sum(self.constraint_coeffs * np.array(ns)), self.target_value):
                valid_mask[ns] = True
        return np.sum(probs[valid_mask])
    
    def batch_compute_metrics(self, result: qt.Result) -> Tuple[np.ndarray, np.ndarray]:
        """Batch compute probs_over_time and satisfaction_rates."""
        probs_over_time = np.zeros((len(self.times),) + (self.N,) * self.num_modes)
        satisfaction_rates = np.zeros(len(self.times))
        for t_idx, rho_t in enumerate(result.states):
            probs_t = self.compute_fock_probs(rho_t)
            probs_over_time[t_idx] = probs_t
            satisfaction_rates[t_idx] = self.compute_satisfaction_rate(probs_t)
        return probs_over_time, satisfaction_rates
    
    def plot_evolution(
        self,
        probs_over_time: np.ndarray,
        satisfaction_rates: np.ndarray,
        plot_states: Optional[List[Tuple[int, ...]]] = None,
        fig_dir: str = 'figs',
        fig_name: str = 'verification_plot.svg'
    ) -> None:
        """Generate and save plots: state probs and satisfaction rate."""
        ensure_dir(fig_dir)
        
        if plot_states is None:
            # Auto-select valid states (e.g., boundaries and samples)
            plot_states = [(0, 0, 6), (0, 6, 0)]  # Defaults to initials
        
        # Reshape plot_states for 3 modes
        if self.num_modes == 3:
            plot_states = [(ns[0], ns[1], ns[2]) for ns in plot_states[:10]]  # Limit for plot
        state_labels = [f"|{' '.join(map(str, ns))}⟩" for ns in plot_states]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_states)))
        state_probs = np.array([[probs_over_time[:, *ns].sum() if len(ns) < self.num_modes else probs_over_time[:, *ns]
                                 for ns in plot_states]])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Selected state probabilities
        for i, (ns, label) in enumerate(zip(plot_states, state_labels)):
            ax1.plot(self.times, state_probs[:, i], label=label, linewidth=2, color=colors[i])
        ax1.set_xlabel('Evolution Time $t$')
        ax1.set_ylabel('Probability $|\\langle \\mathbf{n} | \\psi(t) \\rangle|^2$')
        ax1.set_title('Selected Fock State Probabilities under $H_d$')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Satisfaction rate with max lines
        ax2.plot(self.times, satisfaction_rates, 'r-', linewidth=3, label='Satisfaction Rate')
        for i, (ns, color) in enumerate(zip(plot_states, colors)):
            max_val = np.max(state_probs[10:, i])  # Skip transient
            ax2.hlines(max_val, self.times[0], self.times[-1], colors=color, linestyles='dashed', alpha=0.5)
            ax2.text(self.times[-1], max_val, f"{state_labels[i]} max={max_val:.2f}",
                     fontsize=8, va='bottom', ha='right')
        ax2.set_xlabel('Evolution Time $t$')
        ax2.set_ylabel('Satisfaction Rate')
        ax2.set_title('Constraint Satisfaction Rate (≈1.0)')
        ax2.set_ylim(0, 1.01)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        suptitle = f'Initial Superposition of Boundaries; Constraint: $\\sum c_i n_i = {self.target_value}$'
        fig.suptitle(suptitle, y=0.98)
        plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches='tight')
        plt.show()
    
    def print_summary(self, probs_over_time: np.ndarray, satisfaction_rates: np.ndarray) -> None:
        """Print final satisfaction and top states."""
        print(f"\nFinal satisfaction rate: {satisfaction_rates[-1]:.6f} (should ≈1.0)")
        final_probs = probs_over_time[-1]
        flat_probs = []
        for ns in product(range(self.N), repeat=self.num_modes):
            prob = final_probs[ns]
            if prob > 1e-3:
                constr_val = sum(self.constraint_coeffs * np.array(ns))
                valid = "VALID" if np.isclose(constr_val, self.target_value) else "INVALID"
                flat_probs.append(((ns, prob, valid, constr_val)))
        flat_probs.sort(key=lambda x: x[1], reverse=True)
        print("Top 5 final Fock states by probability:")
        for i, (ns, prob, valid, constr_val) in enumerate(flat_probs[:5]):
            print(f"  |{','.join(map(str, ns))}⟩: {prob:.4f} ({valid}, constr={constr_val:.1f})")

# Example usage
if __name__ == "__main__":
    # Parameters
    N = 7
    constraint_coeffs = [3, 1, 1]
    target_value = 6.0
    null_space_basis = [[1, -3, 0], [0, 1, -1]]
    g = 1.0
    times = np.linspace(0, 100, 200)
    
    # Initialize simulator
    sim = BosonicDriverSimulator(
        N=N, constraint_coeffs=constraint_coeffs, target_value=target_value,
        null_space_basis=null_space_basis, g=g, times=times
    )
    
    # Initial states (superposition of boundaries)
    initial_focks = [[0, 0, 6], [0, 6, 0]]
    psi_0 = sim.create_initial_state(initial_focks, superposition=True)
    
    # Evolve
    result = sim.evolve(psi_0)
    
    # Compute metrics
    probs_over_time, satisfaction_rates = sim.batch_compute_metrics(result)
    
    # Plot and summarize
    sim.plot_evolution(probs_over_time, satisfaction_rates)
    sim.print_summary(probs_over_time, satisfaction_rates)