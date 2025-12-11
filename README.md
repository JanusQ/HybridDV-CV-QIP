# Codip: A Hybrid Continuous-variable and Discrete-variable Quantum System for Scalable and Accurate Integer Programming

**Codip** is a hardware-software co-design framework enabling scalable integer programming on hybrid CV-DV (Continuous-Variable and Discrete-Variable) architectures. By leveraging superconducting cavities for dense integer storage and transmon qubits for non-linear control, Codip bridges the semantic gap between high-level constraints and physical hardware.

This repository contains the reference implementation of the Codip system, including the constraint-preserving intermediate representation (IR), the Hamiltonian-based compiler, and the runtime drift manager.

## Key Features

* **Hybrid CV-DV Architecture**: Implements the "One-mode One-integer" abstraction to reduce memory requirements by 81% compared to standard DV one-hot encoding.
* **Constraint-Preserving IR**: Analytically computes the kernel of the constraint matrix ($Ax=b$) to construct Hamiltonians that confine evolution strictly within the valid subspace, eliminating the need for penalty terms.
* **Drift Manager**: A runtime module capable of detecting and mitigating photon-loss errors using subspace-based error mitigation and phase watchdogs.
* **High Accuracy**: Achieves a $10^2\times$ to $10^5\times$ improvement in success rate compared to state-of-the-art baselines.

## Requirements

* **Python 3.x**
* **QuTiP**: Used for backend simulation of quantum dynamics and Hamiltonian evolution[cite: 498].
* Other dependencies are listed in `pyproject.toml`.

## Repository Structure

The core source code is located in the `cvIP/` directory.

| File | Description |
| :--- | :--- |
| **`cvIP/main.py`** | **Main Entry Point**. The primary solver script that orchestrates the compilation and execution flow. |
| `cvIP/integerCons.py` | **Frontend & IR Translator**. Handles variable definitions, extracts constraint equalities, and generates Kernel Hamiltonians. |
| `cvIP/jointMitigate.py` | **Drift Manager**. Implements active monitoring and subspace-based error mitigation (Dual-State Purification). |
| `cvIP/bosonicQAOA.py` | Core simulation engine for the CV system QAOA. |
| `cvIP/generate_problems.py` | Utility to generate random benchmark subproblems|
| `cvIP/ddqaoa.py` | Implementation of Discrete-Variable (DV) baselines for comparison. |

## Usage

### 1. Environment Setup
Ensure you have the required dependencies installed. You can install the project in editable mode:

```bash
pip install -e .
```
### 2. Run the Codip Solver

Use the main script `main.py` to solve an integer programming instance. [cite_start]This script triggers the compilation and execution workflow[cite: 184].

```bash
python cvIP/main.py
```

*(Note: You may need to pass specific arguments such as problem path or layer depth depending on your configuration within `main.py`.)*

