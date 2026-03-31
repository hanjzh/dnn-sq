using JuMP, KNITRO, MosekTools, LinearAlgebra

# =============================================================================
# UTILITIES
# =============================================================================

# Check if a matrix is PSD (up to tolerance). Returns (is_psd, λ_min, v_min).
# Returns ([], [], []) if Z is not a matrix.
function psd_check(Z; tol=1e-4)
    if !isa(Z, AbstractMatrix)
        return [], [], []
    end
    E = eigen(Symmetric(Z))
    λ_min = minimum(E.values)
    v_min = E.vectors[:, argmin(E.values)]
    return λ_min >= -tol, λ_min, v_min
end

# Cost for BM-factored form: tr(C * Y * Yᵀ)
function compute_cost(Y, cost)
    return tr(cost * Y * transpose(Y))
end

# Cost for lifted (DNN) form: tr(C * X)
function compute_cost_dnn(X, cost)
    return tr(X * cost)
end

# Forward pass returning pre- and post-activation vectors at each layer.
# ReLU is applied at all hidden layers; the final layer is linear.
function forward_pass_with_states(x::Vector{Float64}, h::Vector{Int}, W::Vector{Matrix{Float64}}, b::Vector{Vector{Float64}})
    n_weight_layers = length(h) - 1

    @assert length(W) == n_weight_layers
    @assert length(b) == n_weight_layers

    pre_activations  = Vector{Vector{Float64}}(undef, n_weight_layers)
    post_activations = Vector{Vector{Float64}}(undef, length(h))

    post_activations[1] = x

    for i in 1:n_weight_layers
        z = W[i] * post_activations[i] + b[i]
        pre_activations[i] = z
        post_activations[i+1] = (i < n_weight_layers) ? max.(z, 0.0) : z
    end

    return pre_activations, post_activations
end

# =============================================================================
# COST MATRIX
# =============================================================================

# Build the lifted cost matrix C for output neuron j (layer n+1).
# Encodes: c(X) = tr(C * X) = x⁺_j - x⁻_j (the output neuron value).
function get_cost(n, h, W, j)
    dim_matvar = 2*sum(h) + 2*h[1] + 1
    C = zeros(dim_matvar, dim_matvar)

    i = n + 1  # output layer index
    pos_ind = sum(h[1:i-1]) + j  # positive-split variable index
    neg_ind = sum(h) + sum(h[1:i-1]) + j  # negative-split variable index

    C[pos_ind, end] =  1/2;  C[end, pos_ind] =  1/2
    C[neg_ind, end] = -1/2;  C[end, neg_ind] = -1/2

    return C
end

# =============================================================================
# CONSTRAINT BUILDERS
# =============================================================================

# Build linear and quadratic lifted constraints for input bounds:
#   l_j ≤ x_j ≤ u_j  (for each input neuron j)
# Introduces slack variables for both lower and upper bound inequalities.
function get_input_constraints(n, h, W, dim_matvar, l_bounds, u_bounds)
    all_lin_ai  = []
    all_qc_ai   = []
    all_lin_bi  = []
    all_qc_bi   = []

    for (i, h_i) in enumerate(h[1:1])  # input layer only
        for j in 1:h_i
            lb = l_bounds[j]
            ub = u_bounds[j]

            pos_ind    = sum(h[1:i-1]) + j
            neg_ind    = sum(h) + sum(h[1:i-1]) + j
            lb_sck_ind = 2*sum(h) + sum(h[1:i-1]) + j          # slack for x ≥ lb
            ub_sck_ind = 2*sum(h) + sum(h[1:i-1]) + h[1] + j  # slack for x ≤ ub

            # --- Lower bound: x - lb ≥ 0  →  lifted linear ---
            A_lb_lin = zeros(dim_matvar, dim_matvar)
            A_lb_lin[pos_ind, end]    = -1/2;  A_lb_lin[end, pos_ind]    = -1/2
            A_lb_lin[neg_ind, end]    =  1/2;  A_lb_lin[end, neg_ind]    =  1/2
            A_lb_lin[lb_sck_ind, end] =  1/2;  A_lb_lin[end, lb_sck_ind] =  1/2
            push!(all_lin_ai, A_lb_lin)
            push!(all_lin_bi, -lb)

            # --- Lower bound: quadratic cross-term ---
            A = A_lb_lin[1:end-1, end]
            A_lb_qc = zeros(dim_matvar, dim_matvar)
            A_lb_qc[1:end-1, 1:end-1] .= 4 * (A * A')
            push!(all_qc_ai, A_lb_qc)
            push!(all_qc_bi, lb^2)

            # --- Upper bound: ub - x ≥ 0  →  lifted linear ---
            A_ub_lin = zeros(dim_matvar, dim_matvar)
            A_ub_lin[pos_ind, end]    =  1/2;  A_ub_lin[end, pos_ind]    =  1/2
            A_ub_lin[neg_ind, end]    = -1/2;  A_ub_lin[end, neg_ind]    = -1/2
            A_ub_lin[ub_sck_ind, end] =  1/2;  A_ub_lin[end, ub_sck_ind] =  1/2
            push!(all_lin_ai, A_ub_lin)
            push!(all_lin_bi, ub)

            # --- Upper bound: quadratic cross-term ---
            A = A_ub_lin[1:end-1, end]
            A_ub_qc = zeros(dim_matvar, dim_matvar)
            A_ub_qc[1:end-1, 1:end-1] .= 4 * (A * A')
            push!(all_qc_ai, A_ub_qc)
            push!(all_qc_bi, ub^2)
        end
    end

    return all_lin_ai, all_qc_ai, all_lin_bi, all_qc_bi
end

# Build linear and quadratic lifted constraints encoding the affine map between
# consecutive layers: x⁺_{i+1} - x⁻_{i+1} = W[i] x⁺_i + b[i]
function get_consistency_constraints(n, h, W, b, dim_matvar)
    all_lin_ai = []
    all_qc_ai  = []
    all_lin_bi = []
    all_qc_bi  = []

    for (i, h_i) in enumerate(h[1:end-1])  # all layers except output
        h_next       = h[i+1]
        pos_ind      = sum(h[1:i-1]) + 1
        pos_ind_next = sum(h[1:i]) + 1
        neg_ind      = sum(h) + sum(h[1:i-1]) + 1
        neg_ind_next = sum(h) + sum(h[1:i]) + 1

        # Construct A such that: A z = 0 encodes the affine map residual
        A = zeros(h_next, dim_matvar - 1)
        A[:, pos_ind:pos_ind + h_i - 1] .= -W[i]

        if i == 1  # input layer: x = x⁺ - x⁻
            A[:, neg_ind:neg_ind + h_i - 1] .= W[i]
        end

        A[:, pos_ind_next:pos_ind_next + h_next - 1] .=  Matrix{Float64}(I, h_next, h_next)
        A[:, neg_ind_next:neg_ind_next + h_next - 1] .= -Matrix{Float64}(I, h_next, h_next)

        for j in 1:h_next
            A_row = reshape(A[j, :], 1, :)
            A_lin = zeros(dim_matvar, dim_matvar)
            A_lin[1:end-1, end] .= 0.5 * vec(A_row)
            A_lin[end, 1:end-1] .= 0.5 * A_row[:]
            push!(all_lin_ai, A_lin)
            push!(all_lin_bi, b[i][j])

            # Quadratic cross-term for this consistency constraint
            A_qc = zeros(dim_matvar, dim_matvar)
            A_qc[1:end-1, 1:end-1] .= 4 * (A_lin[1:end-1, end] * A_lin[1:end-1, end]')
            push!(all_qc_ai, A_qc)
            push!(all_qc_bi, b[i][j]^2)
        end
    end

    return all_lin_ai, all_qc_ai, all_lin_bi, all_qc_bi
end

# Build lifted equality constraints for ReLU complementarity at hidden neurons:
#   x⁺_j * x⁻_j = 0  (encoded as tr(A X) = 0)
function get_relu_constraints(n, h, W, dim_matvar)
    all_ai = []
    all_bi = []

    for (i, h_i) in enumerate(h)
        if i == 1 || i == length(h)  # skip input and output layers
            continue
        end
        pos_ind = sum(h[1:i-1])
        neg_ind = sum(h) + sum(h[1:i-1])

        for j in 1:h_i
            A = zeros(dim_matvar, dim_matvar)
            A[pos_ind+j, neg_ind+j] = 1/2
            A[neg_ind+j, pos_ind+j] = 1/2
            push!(all_ai, A)
            push!(all_bi, 0)
        end
    end

    return all_ai, all_bi
end

# Homogenization constraint: z_{end} = 1  →  lifted as tr(A X) = 1
function get_homog_constraint(n, h, W, dim_matvar)
    A = zeros(dim_matvar, dim_matvar)
    A[end, end] = 1
    return A, 1
end

# Build elementwise non-negativity constraints for the upper triangle of an
# n×n PSD matrix: -X[i,j] ≤ 0  (symmetric form, so off-diagonal entries
# get weight 1/2 on both sides).
function get_quad_elem_non_neg_constraints(n::Int)
    constraints_A = []
    constraints_b = []
    for i in 1:n
        for j in i:n
            A = zeros(n, n)
            if i == j
                A[i, j] = -1.0
            else
                A[i, j] = -0.5
                A[j, i] = -0.5
            end
            push!(constraints_A, A)
            push!(constraints_b, 0)
        end
    end
    return constraints_A, constraints_b
end

# =============================================================================
# FEASIBLE POINT INITIALIZATION
# =============================================================================

# Construct a DNN-feasible point from a concrete input x_sample via forward
# pass, building the positive/negative split variables and bound slacks.
function get_feas_pt(x_sample, h, W, b, l_in, u_in, all_constraints_ai, all_constraints_bi)
    h_0 = h[1]
    pre_activ, post_activ = forward_pass_with_states(x_sample, h, W, b)

    lambda_plus = max.(x_sample, 0.0)
    lambda_neg  = -min.(x_sample, 0.0)
    popfirst!(post_activ)  # remove input layer (already captured above)

    final_pre  = pre_activ[end]
    final_post = post_activ[end]

    if !isempty(post_activ[1:end-1])  # at least one hidden layer
        hidden_post = reduce(vcat, post_activ[1:end-1])
        hidden_pre  = reduce(vcat, pre_activ[1:end-1])
        lambda_plus = vcat(lambda_plus, hidden_post)
        lambda_neg  = vcat(lambda_neg,  hidden_post - hidden_pre)
    end

    lambda_plus = vcat(lambda_plus, max.(final_post, 0.0))
    lambda_neg  = vcat(lambda_neg,  -min.(final_post, 0.0))

    # Slack variables satisfying: x - lb ≥ 0 and ub - x ≥ 0
    slack_lb = lambda_plus[1:h_0] - lambda_neg[1:h_0] - l_in
    slack_ub = u_in - lambda_plus[1:h_0] + lambda_neg[1:h_0]

    return vcat(lambda_plus, lambda_neg, slack_lb, slack_ub, [1])
end

# Wrap get_feas_pt to also build the lifted matrix X = zz' and factor Y.
function get_feas_pt_dnn(r, feas_pt_init, cost_cpp, x_sample, h, W, b, l_in, u_in, all_constraints_ai, all_constraints_bi)
    if !feas_pt_init
        return [], [], []
    end

    feas_pt    = get_feas_pt(x_sample, h, W, b, l_in, u_in, all_constraints_ai, all_constraints_bi)
    dim        = size(cost_cpp, 1)
    X_init     = feas_pt * feas_pt'
    Y_init     = zeros(dim, r)
    Y_init[:, 1] .= feas_pt

    return feas_pt, X_init, Y_init
end

# =============================================================================
# CONSTRAINT ASSEMBLY
# =============================================================================

# Assemble all DNN equality constraints from the four building blocks.
function get_dnn_constraints(n, h, W, b, l_in, u_in)
    dim_matvar = 2*sum(h) + 2*h[1] + 1

    all_ai = []
    all_bi = []

    # Input bound constraints (linear + quadratic cross-terms)
    lin_in_ai, qc_in_ai, lin_in_bi, qc_in_bi = get_input_constraints(n, h, W, dim_matvar, l_in, u_in)
    all_ai = vcat(all_ai, lin_in_ai, qc_in_ai)
    all_bi = vcat(all_bi, lin_in_bi, qc_in_bi)

    # Affine map consistency constraints (linear + quadratic cross-terms)
    lin_con_ai, qc_con_ai, lin_con_bi, qc_con_bi = get_consistency_constraints(n, h, W, b, dim_matvar)
    all_ai = vcat(all_ai, lin_con_ai, qc_con_ai)
    all_bi = vcat(all_bi, lin_con_bi, qc_con_bi)

    # ReLU complementarity constraints
    relu_ai, relu_bi = get_relu_constraints(n, h, W, dim_matvar)
    all_ai = vcat(all_ai, relu_ai)
    all_bi = vcat(all_bi, relu_bi)

    # Homogenization constraint: z_{end} = 1
    homog_mat, homog_b = get_homog_constraint(n, h, W, dim_matvar)
    all_ai = vcat(all_ai, [homog_mat])
    all_bi = vcat(all_bi, [homog_b])

    return all_ai, all_bi
end

# Add DNN (lifted) equality constraints: tr(A X) = b
function add_all_constraints_dnn(model, all_constraints_ai, all_constraints_bi, X)
    prob_constraints = []
    for (ind, A) in enumerate(all_constraints_ai)
        cref = @constraint(model, tr(A*X) == all_constraints_bi[ind], base_name="eq$ind")
        push!(prob_constraints, cref)
    end
    return prob_constraints
end

# Add BM (factored) equality constraints: tr(A Y Yᵀ) = b
function add_all_constraints(model, all_constraints_ai, all_constraints_bi, Y)
    prob_constraints = []
    for (ind, A) in enumerate(all_constraints_ai)
        cref = @constraint(model, tr(A*Y*transpose(Y)) == all_constraints_bi[ind], base_name="eq$ind")
        push!(prob_constraints, cref)
    end
    return prob_constraints
end

# =============================================================================
# SOLVERS
# =============================================================================

# Solve the convex DNN relaxation via Mosek interior-point.
# Returns (X_opt, solve_time, termination_status).
function dnn_relaxation_solve(which_bound, dim_matvar, all_constraints_ai, all_constraints_bi, cost; X_start=nothing)
    model = Model(Mosek.Optimizer)

    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS",   1e-6)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS",   1e-6)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-10)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED",  1e-10)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_INFEAS",     1e-10)
    set_optimizer_attribute(model, "MSK_IPAR_INTPNT_MAX_ITERATIONS", 5000)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS",           1)
    set_optimizer_attribute(model, "MSK_IPAR_PRESOLVE_USE",          0)

    @variable(model, X[1:dim_matvar, 1:dim_matvar], PSD)

    if X_start !== nothing && !isempty(X_start)
        for i in 1:dim_matvar, j in 1:dim_matvar
            set_start_value(X[i, j], X_start[i, j])
        end
        set_optimizer_attribute(model, "MSK_IPAR_INTPNT_STARTING_POINT", 1)
    end

    set_silent(model)

    constraint_data = []

    # Elementwise non-negativity constraints: X[i,j] ≥ 0
    ineq_ai, ineq_bi = get_quad_elem_non_neg_constraints(dim_matvar)
    for (ind, A) in enumerate(ineq_ai)
        cref = @constraint(model, tr(A*X) <= ineq_bi[ind], base_name="ineq$ind")
        push!(constraint_data, (cref, A, ineq_bi[ind], :leq))
    end

    # DNN equality constraints
    for (ind, A) in enumerate(all_constraints_ai)
        cref = @constraint(model, tr(A*X) == all_constraints_bi[ind], base_name="eq$ind")
        push!(constraint_data, (cref, A, all_constraints_bi[ind], :eq))
    end

    if which_bound == 1
        @objective(model, Min,  tr(X*cost))
    elseif which_bound == 2
        @objective(model, Min, -tr(X*cost))
    end

    optimize!(model)

    if !(termination_status(model) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED] ||
         primal_status(model) == MOI.FEASIBLE_POINT)
        println("DNN solver did not find a feasible solution.")
        return NaN, NaN, NaN
    end

    return value.(X), JuMP.solve_time(model), termination_status(model)
end

# Solve the BM-factored DNN relaxation via KNITRO (nonconvex NLP).
# Returns (Y_opt, solve_time, status, S_cert_naive, S_cert_ours, eigenmax_time, eigenmax_status).
function bm_dnn_relaxation_solve(which_bound, r, h, all_constraints_ai, all_constraints_bi, cost, logs_dir; Y_start=nothing)
    dim_matvar = 2*sum(h) + 2*h[1] + 1

    model = Model(KNITRO.Optimizer)

    # Solver settings
    set_optimizer_attribute(model, "maxtime",        300.0)
    set_optimizer_attribute(model, "numthreads",     32)
    set_optimizer_attribute(model, "bar_feasible",   0)    # allow infeasible iterates
    set_optimizer_attribute(model, "nlp_algorithm",  1)    # Interior/Direct
    set_optimizer_attribute(model, "strat_warm_start", 1)
    set_optimizer_attribute(model, "convex",         0)
    set_optimizer_attribute(model, "scale",          2)    # user-provided scaling
    set_optimizer_attribute(model, "soltype",        0)    # require KKT convergence
    set_optimizer_attribute(model, "honorbnds",      0)
    set_optimizer_attribute(model, "presolve",       0)
    set_optimizer_attribute(model, "bar_murule",     4)    # dampmpc
    set_optimizer_attribute(model, "bar_penaltycons", 0)
    set_optimizer_attribute(model, "bar_penaltyrule", 2)   # flex
    set_optimizer_attribute(model, "datacheck",      0)
    set_optimizer_attribute(model, "hessopt",        1)    # exact Hessian
    set_optimizer_attribute(model, "gradopt",        1)    # exact gradients
    set_optimizer_attribute(model, "maxfevals",      1e6)
    set_optimizer_attribute(model, "maxit",          1e6)
    set_optimizer_attribute(model, "xtol",           1e-10)
    set_optimizer_attribute(model, "feastol",        1e-6)
    set_optimizer_attribute(model, "opttol",         1e-6)
    set_optimizer_attribute(model, "feastol_abs",    1e-3)
    set_optimizer_attribute(model, "opttol_abs",     1e-3)

    @variable(model, Y[1:dim_matvar, 1:r])

    if Y_start !== nothing && !isempty(Y_start)
        for i in 1:dim_matvar, j in 1:r
            set_start_value(Y[i, j], Y_start[i, j])
        end
    end

    set_silent(model)

    constraint_data = []

    # Elementwise non-negativity constraints: (Y Yᵀ)[i,j] ≥ 0
    ineq_ai, ineq_bi = get_quad_elem_non_neg_constraints(dim_matvar)
    for (ind, A) in enumerate(ineq_ai)
        cref = @constraint(model, tr(A*Y*transpose(Y)) <= ineq_bi[ind], base_name="ineq$ind")
        push!(constraint_data, (cref, A, ineq_bi[ind], :leq))
    end

    # DNN equality constraints (in BM form)
    for (ind, A) in enumerate(all_constraints_ai)
        cref = @constraint(model, tr(A*Y*transpose(Y)) == all_constraints_bi[ind], base_name="eq$ind")
        push!(constraint_data, (cref, A, all_constraints_bi[ind], :eq))
    end

    if which_bound == 1
        @objective(model, Min,  tr(Y*transpose(Y)*cost))
    elseif which_bound == 2
        @objective(model, Min, -tr(Y*transpose(Y)*cost))
    end

    optimize!(model)

    if !(termination_status(model) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED] ||
         primal_status(model) == MOI.FEASIBLE_POINT)
        println("BM-DNN solver did not find a feasible solution.")
        return NaN, NaN, JuMP.solve_time(model), termination_status(model), NaN, NaN, NaN, NaN
    end

    soln_Y = value.(Y)

    # --- Global optimality certificate ---
    S_cert_naive = build_certificate(which_bound, constraint_data, cost, soln_Y)
    λ_min_opt, S_cert_ours, _, eigenmax_time, eigenmax_status = eigenvalue_max_sdp(which_bound, constraint_data, cost, soln_Y)

    return value.(Y), JuMP.solve_time(model), termination_status(model),
           S_cert_naive, S_cert_ours, eigenmax_time, eigenmax_status
end

# =============================================================================
# GLOBAL OPTIMALITY CERTIFICATE
# =============================================================================

# Construct the naive KKT certificate matrix S = ∇f - Σ λᵢ Aᵢ
# using Lagrange multipliers returned directly by the NLP solver.
# If S ⪰ 0, the BM solution Z = YYᵀ is globally optimal for the DNN.
function build_certificate(which_bound, constraint_data, cost, sol, tol=1e-4)
    S_cert = (which_bound == 1) ? copy(cost) : -copy(cost)

    for (cref, A, b, sense) in constraint_data
        λ = JuMP.dual(cref)

        if sense == :leq
            slack = b - tr(A * sol * transpose(sol))
            if abs(λ * slack) > tol
                @warn "Complementary slackness violated: λ=$λ, slack=$slack"
            end
        end

        S_cert .-= λ * A
    end

    return S_cert
end

# Search for certificate.
# Solve the eigenvalue maximization SDP to find the best Lagrange multipliers
# maximizing λ_min(S(λ)).  If the optimum t* ≥ 0, the BM solution is certified.
# Returns (t_min, S_cert_opt, λ_opt, solve_time, status).
function eigenvalue_max_sdp(which_bound, constraint_data, cost, sol)
    S_cert_naive = build_certificate(which_bound, constraint_data, cost, sol)

    model = Model(Mosek.Optimizer)

    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS",    1e-8)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS",    1e-8)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",  1e-8)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED",   1e-7)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL", 1e10)
    set_optimizer_attribute(model, "MSK_IPAR_INTPNT_MAX_ITERATIONS",  1000)
    set_optimizer_attribute(model, "MSK_IPAR_INTPNT_SOLVE_FORM",      1)
    set_silent(model)

    n = size(cost, 1)
    Id = Matrix{Float64}(I, n, n)

    @variable(model, S_cert[1:n, 1:n], Symmetric)
    @variable(model, t)
    @variable(model, λ[1:length(constraint_data)])

    @objective(model, Max, t)

    # S_cert ⪰ t I
    @constraint(model, S_cert - t*Id in PSDCone())

    # Warm-start λ from naive multipliers
    for (idx, (cref, A, b, sense)) in enumerate(constraint_data)
        set_start_value(λ[idx], JuMP.dual(cref))
    end

    # Warm-start S_cert from naive certificate
    if isa(S_cert_naive, AbstractMatrix)
        for i in 1:n, j in 1:n
            set_start_value(S_cert[i, j], S_cert_naive[i, j])
        end
        set_optimizer_attribute(model, "MSK_IPAR_INTPNT_STARTING_POINT", 1)
    end

    # KKT stationarity: S_cert + Σ λᵢ Aᵢ = ∇f
    grad_f = (which_bound == 1) ? copy(cost) : -copy(cost)
    for i in 1:n, j in 1:n
        if i <= j
            @constraint(model,
                S_cert[i,j] + sum(λ[idx] * A[i,j] for (idx, (cref, A, b, sense)) in enumerate(constraint_data)) == grad_f[i,j]
            )
        end
    end

    # ‖S Y‖₁ ≤ ‖S_naive Y‖₁  (stationarity in the BM tangent space)
    n_mat, r_mat = size(sol)
    grad_norm = sum(abs.(S_cert_naive * sol))
    SY_vec = [sum(S_cert[i,k] * sol[k,j] for k in 1:n_mat) for i in 1:n_mat, j in 1:r_mat]
    @constraint(model, [grad_norm; vec(SY_vec)] in MOI.NormOneCone(1 + n_mat*r_mat))

    # Complementary slackness: λᵢ ≤ 0 and λᵢ * sᵢ ≈ 0 for inequality constraints
    total      = AffExpr[]
    total_naive = Float64[]
    for (idx, (cref, A, b, sense)) in enumerate(constraint_data)
        if sense == :leq
            λ_naive = JuMP.dual(cref)
            slack   = b - tr(A * sol * transpose(sol))
            @constraint(model, λ[idx] <= 0)
            push!(total_naive, slack * λ_naive)
            push!(total,       slack * λ[idx])
        end
        t_naive = sum(abs.(total_naive))
        @constraint(model, [t_naive; total] in MOI.NormOneCone(1 + length(total)))
    end

    optimize!(model)

    return objective_value(model), value.(S_cert), value.(λ), JuMP.solve_time(model), termination_status(model)
end
