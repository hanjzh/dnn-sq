using NeuralVerification

include("export_2_nnet.jl")

module MILP_SOLVE
    include("milp.jl")
end

module SDP_RELAX
    include("sdp_relax.jl")
end

module DNN_RELAX
    include("dnn_relax.jl")
end

# ─── Problem configuration ────────────────────────────────────────────────────

p             = 2    # BM rank
lb_const      = -1   # input lower bound (uniform across all inputs)
ub_const      = 0.1  # input upper bound (uniform across all inputs)
which_bound   = 1    # 1: minimize output, 2: maximize output
output_index  = 1    # output neuron to extremize (1-indexed)
feas_pt_init  = true # true: warm-start from the midpoint of the input set

# ─── Network architecture ─────────────────────────────────────────────────────

h_0       = 2          # input dimension
h_n       = 1          # output dimension
n         = 2          # number of hidden layers
h         = [h_0, 10, h_n]
nnet_name = "brown.nnet"

# ─── Generate network and input set ──────────────────────────────────────────

logs_dir = "eval/logs_$nnet_name"
mkpath(logs_dir)
nnet_loc = joinpath(logs_dir, nnet_name)

W, b, N, mean_vals, range_vals = generate_nn(h_0, h_n, n, h, nnet_name)
write_nnet_full(nnet_loc, h, W, b)

# Random input set within [lb_const, ub_const]
r1   = lb_const .+ (ub_const - lb_const) .* rand(h_0)
r2   = lb_const .+ (ub_const - lb_const) .* rand(h_0)
l_in = min.(r1, r2)
u_in = max.(r1, r2)

println("input bounds: lb = $lb_const, ub = $ub_const")
println("output_index = $output_index, which_bound = $which_bound")

# ─── STEP 0: MILP (ground truth) ─────────────────────────────────────────────

println("=" ^ 60)
println("STEP 0: Solve MILP (ground truth via MIPVerify)")
println("=" ^ 60)
milp_cost, milp_full_soln, milp_time, milp_status = MILP_SOLVE.solve_milp(which_bound, nnet_loc, output_index, l_in, u_in)
println("  Bound:      $milp_cost")

# ─── STEP 1: Feasible point (warm-start for relaxations) ─────────────────────

println("=" ^ 60)
println("STEP 1: Feasible point initialization (midpoint of input set)")
println("=" ^ 60)
x_sample = l_in .+ (u_in .- l_in) / 2
println("  x_sample: $x_sample")

# ─── STEP 2: c-SDP relaxation ────────────────────────────────────────────────

println("\n", "=" ^ 60)
println("STEP 2: Solve canonical SDP relaxation (Mosek interior-point)")
println("=" ^ 60)
cost_sdp = SDP_RELAX.get_cost(h, output_index)
all_constraints_ai_eq, all_constraints_bi_eq, all_constraints_ai_ineq, all_constraints_bi_ineq =
    SDP_RELAX.get_sdp_constraints(n, h, W, b, l_in, u_in)
_, X_init_sdp, _ = SDP_RELAX.get_feas_pt_sdp(p, feas_pt_init, cost_sdp, x_sample, h, W, b,
    all_constraints_ai_eq, all_constraints_bi_eq, all_constraints_ai_ineq, all_constraints_bi_ineq)
X_sdp, sdp_time, sdp_status = SDP_RELAX.sdp_relaxation_solve(which_bound,
    all_constraints_ai_ineq, all_constraints_bi_ineq,
    all_constraints_ai_eq,   all_constraints_bi_eq,
    cost_sdp, logs_dir; X_start=X_init_sdp)
sdp_cost = SDP_RELAX.compute_cost(X_sdp, cost_sdp)
println("  Status:     $sdp_status")
println("  Bound:      $sdp_cost")
println("  Solve time: $(round(sdp_time, digits=4))s")
println("  (globally optimal — convex SDP solved to interior-point tolerance)")

# ─── STEP 3: DNN relaxation ──────────────────────────────────────────────────

println("\n", "=" ^ 60)
println("STEP 3: Solve DNN relaxation (Mosek interior-point)")
println("=" ^ 60)
dim_matvar = 2*sum(h) + 2*h[1] + 1
cost_dnn   = DNN_RELAX.get_cost(n, h, W, output_index)
all_constraints_ai, all_constraints_bi = DNN_RELAX.get_dnn_constraints(n, h, W, b, l_in, u_in)
_, X_init_dnn, Y_init = DNN_RELAX.get_feas_pt_dnn(p, feas_pt_init, cost_dnn, x_sample, h, W, b, l_in, u_in,
    all_constraints_ai, all_constraints_bi)
X_dnn, dnn_time, dnn_status = DNN_RELAX.dnn_relaxation_solve(which_bound, dim_matvar,
    all_constraints_ai, all_constraints_bi, cost_dnn; X_start=X_init_dnn)
dnn_cost = DNN_RELAX.compute_cost_dnn(X_dnn, cost_dnn)
println("  Status:     $dnn_status")
println("  Bound:      $dnn_cost")
println("  Solve time: $(round(dnn_time, digits=4))s")
println("  (globally optimal — convex SDP solved to interior-point tolerance)")

# ─── STEP 4: BM-DNN relaxation ───────────────────────────────────────────────

println("\n", "=" ^ 60)
println("STEP 4: Solve BM-factored DNN (KNITRO NLP, rank p=$p)")
println("=" ^ 60)
println("  BM factorization X = YYᵀ converts the convex DNN into a nonconvex NLP.")
println("  KNITRO finds a KKT point — may be a local (not global) optimum.")
Y_bm, bm_time, bm_status, S_cert_naive, S_cert_ours, eigenmax_time, eigenmax_status =
    DNN_RELAX.bm_dnn_relaxation_solve(which_bound, p, h, all_constraints_ai, all_constraints_bi,
        cost_dnn, logs_dir; Y_start=Y_init)
bm_cost = DNN_RELAX.compute_cost(Y_bm, cost_dnn)
println("  Status:     $bm_status")
println("  Bound:      $bm_cost")
println("  Solve time: $(round(bm_time, digits=4))s")

# ─── STEP 5: Global optimality certificate ───────────────────────────────────

println("\n", "=" ^ 60)
println("STEP 5: Certificate — is the BM-DNN solution globally optimal for the DNN?")
println("=" ^ 60)
println("  Check: does there exist λ such that S(λ) = ∇f - Σλᵢ Aᵢ ⪰ 0?")
println("  λ_min(S) ≥ 0  →  CERTIFIED globally optimal")
println("  λ_min(S) < 0  →  INCONCLUSIVE (multipliers could not confirm it)")

is_psd_naive, λ_naive, _ = DNN_RELAX.psd_check(S_cert_naive)
is_psd_ours,  λ_min,   _ = DNN_RELAX.psd_check(S_cert_ours)

println("\n  --- Naive certificate (NLP solver multipliers) ---")
println("  λ_min(S): $λ_naive")
println("  Result:   $(is_psd_naive ? "✓ CERTIFIED" : "✗ NOT CERTIFIED — LICQ may be violated")")

println("\n  --- Our certificate (eigenvalue maximization over multiplier space) ---")
println("  λ_min(S): $λ_min")
println("  Eigenmax solve time: $(round(eigenmax_time, digits=4))s ($eigenmax_status)")
println("  Result:   $(is_psd_ours ? "✓ CERTIFIED" : "✗ NOT CERTIFIED — no valid multipliers found")")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────

println("\n", "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("  MILP (exact):     $milp_cost")
println("  c-SDP bound:      $sdp_cost  (gap: $(abs(milp_cost - sdp_cost)))")
println("  DNN bound:        $dnn_cost  (gap: $(abs(milp_cost - dnn_cost)))")
println("  BM-DNN bound:     $bm_cost   (gap: $(abs(milp_cost - bm_cost)))")
println("  BM-DNN certified: $(is_psd_ours ? "YES ✓" : "NO ✗")")