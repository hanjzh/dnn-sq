# Running `main.jl`

`main.jl` runs a full neural network verification pipeline on a randomly generated
two-hidden-layer ReLU network. It solves four progressively tighter formulations —
mixed-integer linear programming (MILP) [1], the canonical semidefinite relaxation
(c-SDP) [2], the tighter doubly non-negative relaxation (DNN) [3], and our proposed 
Burer-Monteiro factored DNN (BM-DNN) — then checks whether the BM-DNN solution
is globally optimal via our search for a certificate over the space of non-unique 
Lagrange multipliers. See our CDC paper for more technical details: [LINK PAPER HERE]

### References

[1] V. Tjeng, K. Xiao, and R. Tedrake, "Evaluating robustness of neural networks with mixed integer programming," in *Int. Conf. on Learning Representations (ICLR)*, 2019. [arXiv:1711.07356](https://arxiv.org/abs/1711.07356)

[2] A. Raghunathan, J. Steinhardt, and P. Liang, "Semidefinite relaxations for certifying robustness to adversarial examples," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, pp. 10900–10910, 2018. [arXiv:1811.01057](https://arxiv.org/abs/1811.01057)

[3] R. A. Brown, E. Schmerling, N. Azizan, and M. Pavone, "A unified view of SDP-based neural network verification through completely positive programming," in *Proc. Int. Conf. on Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 151, pp. 9334–9355, 2022. [arXiv:2203.03034](https://arxiv.org/abs/2203.03034)

---

## Prerequisites

**Julia ≥ 1.9** is required. Download from [julialang.org/downloads](https://julialang.org/downloads/).

Two of the solvers used are **commercial software that require a separate installation
and a license** before the Julia packages will work. Both offer free academic licenses.
See the solver sections below before running `Pkg.instantiate()`.

---

## File structure

```
project/
├── main.jl              # entry point
├── dnn_relax.jl         # DNN and BM-DNN solvers, certificate
├── sdp_relax.jl         # canonical SDP solver
├── milp.jl              # MILP solver (MIPVerify)
├── export_2_nnet.jl     # network generation and .nnet I/O
├── Project.toml         # Julia package manifest
└── Manifest.toml        # pinned dependency versions
```

Output is written to `eval/logs_brown.nnet/` (created automatically).

---

## Step 1 — Install Mosek (required for SDP, DNN, BM-DNN certificate)

Mosek is used by `sdp_relax.jl` and `dnn_relax.jl` via `MosekTools.jl`. You need
both the Mosek solver binaries and a license file.

**1a. Get a license**

Academic licenses are free at [mosek.com/products/academic-licenses](https://www.mosek.com/products/academic-licenses/).
Fill in the form with your university email. Mosek will email you a `mosek.lic` file.

Place the license file at:
```
~/mosek/mosek.lic          # Linux / macOS
%USERPROFILE%\mosek\mosek.lic   # Windows
```

Alternatively, set the environment variable `MOSEKLM_LICENSE_FILE` to the full path
of the license file before launching Julia.

**1b. Install the solver binaries**

`MosekTools.jl` downloads the Mosek binaries automatically when the package is first
built, so no separate binary installation is needed. The Julia package takes care of it.

---

## Step 2 — Install KNITRO (required for BM-DNN)

KNITRO is used by `dnn_relax.jl` via `KNITRO.jl`. It requires the Artelys KNITRO
solver to be installed on your system separately from Julia.

**2a. Get a license**

Academic licenses are available at [artelys.com/solvers/knitro/academic](https://www.artelys.com/solvers/knitro/academic/).
Fill in the request form. Artelys will provide a license file (`artelys_lic` or
similar) and download instructions.

**2b. Install the solver binaries**

Download the KNITRO package for your platform from the Artelys portal after your
license is approved. Follow their installation guide; it typically results in a
directory like `/opt/artelys/knitro-14.x.x/`.

**2c. Set environment variables**

Before launching Julia, tell `KNITRO.jl` where the binaries live:

```bash
# Linux
export KNITRODIR=/opt/artelys/knitro-14.x.x
export LD_LIBRARY_PATH=$KNITRODIR/lib:$LD_LIBRARY_PATH

# macOS
export KNITRODIR=/opt/artelys/knitro-14.x.x
export DYLD_LIBRARY_PATH=$KNITRODIR/lib:$DYLD_LIBRARY_PATH

# Windows (PowerShell)
$env:KNITRODIR = "C:\artelys\knitro-14.x.x"
$env:PATH = "$env:KNITRODIR\lib;$env:PATH"
```

You may want to add these to your shell profile (`.bashrc`, `.zshrc`, etc.) so they
persist across sessions.

---

## Step 3 — Install Julia packages

With Julia installed and the two solvers above in place, activate the project
environment and install all remaining packages:

```julia
julia --project=.

julia> using Pkg
julia> Pkg.instantiate()
```

`Pkg.instantiate()` reads `Manifest.toml` and installs the exact pinned versions of
every dependency. This includes all free packages automatically. The first run will
also trigger the Mosek binary download.

If you prefer to install without the pinned manifest (e.g. to get latest versions):

```julia
julia> Pkg.add(["JuMP", "MosekTools", "KNITRO", "NeuralVerification",
                "MIPVerify", "GLPK", "LazySets", "LinearAlgebra"])
```

The full dependency list for `main.jl` and its includes is:

| Package            | Source                    | Notes                                    |
|--------------------|---------------------------|------------------------------------------|
| `JuMP`             | Julia registry            | Optimization modelling layer             |
| `MathOptInterface` | Julia registry            | Pulled in automatically by JuMP          |
| `MosekTools`       | Julia registry            | Requires Mosek license (Step 1)          |
| `KNITRO`           | Julia registry            | Requires KNITRO install + license (Step 2) |
| `NeuralVerification` | Julia registry          | Network parsing and MILP encoding        |
| `MIPVerify`        | Julia registry            | Mixed-integer encoding of ReLU networks  |
| `GLPK`             | Julia registry            | LP solver used internally by MIPVerify   |
| `LazySets`         | Julia registry            | Hyperrectangle input set definition      |
| `LinearAlgebra`    | Julia stdlib              | Built-in, no install needed              |
| `Printf`, `CSV`, `Random` | Julia stdlib / registry | Built-in or auto-installed      |

The `Project.toml` also lists several packages used by other scripts in the project
(e.g. `StatsPlots`, `PGFPlotsX`, `Clarabel`) that are not needed to run `main.jl`
specifically. They will be installed by `Pkg.instantiate()` regardless.

---

## Step 4 — Configure `main.jl`

The top of `main.jl` contains all user-facing parameters:

```julia
p             = 2    # Burer-Monteiro rank (try p = 2 or 3 to start)
lb_const      = -1   # input lower bound, applied uniformly to all inputs
ub_const      = 0.1  # input upper bound, applied uniformly to all inputs
which_bound   = 1    # 1 = minimize output neuron, 2 = maximize
output_index  = 1    # which output neuron to bound (1-indexed)
feas_pt_init  = true # warm-start solvers from the input-set midpoint

h_0       = 2        # input dimension
h_n       = 1        # output dimension
n         = 2        # number of hidden layers
h         = [h_0, 10, h_n]  # layer sizes: input, hidden..., output
nnet_name = "brown.nnet"    # filename used when saving the network
```

The network is randomly generated on each run using a fixed random seed (seed `1`
by default, set in `generate_nn`). To use a different network, either change the
seed via `network_weights_bound_rng_ind` in `generate_nn`, modify `h` for a different
architecture, or replace the generation block with a call to `extract_network_data`
on an existing `.nnet` file.

---

## Step 5 — Run

From the project root:

```bash
julia --project=. main.jl
```

Or from inside the Julia REPL:

```julia
julia> include("main.jl")
```

---

## Expected output

```
input bounds: lb = -1, ub = 0.1
output_index = 1, which_bound = 1
============================================================
STEP 0: Solve MILP (ground truth via MIPVerify)
============================================================
  Bound:      <value>
============================================================
STEP 1: Feasible point initialization (midpoint of input set)
============================================================
  x_sample: [...]
...
============================================================
SUMMARY
============================================================
  MILP (exact):     <value>
  c-SDP bound:      <value>  (gap: <value>)
  DNN bound:        <value>  (gap: <value>)
  BM-DNN bound:     <value>  (gap: <value>)
  BM-DNN certified: YES ✓ / NO ✗
```

The DNN bound will always be ≤ the c-SDP bound (tighter relaxation). The BM-DNN
bound will match the DNN bound when the BM factorization recovers the global optimum
of the DNN relaxation, which the certificate step confirms.

---

## Troubleshooting

**`KNITRO.jl` fails to load / "library not found"**
The `KNITRODIR` and library path environment variables must be set *before* Julia
starts. Set them in your shell profile and open a new terminal, or pass them
explicitly: `KNITRODIR=/path/to/knitro julia --project=. main.jl`.

**`MosekTools` throws a license error on first use**
Check that `~/mosek/mosek.lic` exists and is readable, or that
`MOSEKLM_LICENSE_FILE` points to it. The license file must not be expired.

**`NeuralVerification` or `MIPVerify` not found by `Pkg.instantiate()`**
These packages are in the general Julia registry. If `instantiate` fails, try:
```julia
julia> Pkg.add("NeuralVerification")
julia> Pkg.add("MIPVerify")
```

**Slow first run**
Julia compiles packages on first use. The second run of the same session (or after
precompilation) will be significantly faster.
