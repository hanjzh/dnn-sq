using JuMP, LinearAlgebra, MosekTools, Random

# =============================================================================
# FEASIBLE POINT INITIALIZATION
# =============================================================================

# Build a feasible point for the SDP by running a forward pass through the
# network, then construct the lifted X = zzᵀ and an initial factor Y.
function get_feas_pt_sdp(r, feas_pt_init, cost_direct, x_sample, h, W, b, all_constraints_ai_eq, all_constraints_bi_eq, all_constraints_ai_ineq, all_constraints_bi_ineq)
    if !feas_pt_init
        return [], [], []
    end

    dim_matvar  = size(cost_direct, 1)
    feas_pt_sdp = SDP_RELAX.get_feas_pt(x_sample, h, W, b, all_constraints_ai_eq, all_constraints_bi_eq, all_constraints_ai_ineq, all_constraints_bi_ineq)
    X_init      = feas_pt_sdp * feas_pt_sdp'

    # Spread the feasible direction across r columns via a random Stiefel element
    A      = randn(Random.default_rng(), r, r)
    Q, R   = qr(A)
    Q_mat  = Matrix(Q) * Diagonal(sign.(diag(R)))  # fix sign ambiguity

    Y_init      = zeros(dim_matvar, r)
    Y_init[:, 1] = feas_pt_sdp
    Y_init       = Y_init * Q_mat

    return feas_pt_sdp, X_init, Y_init
end

# =============================================================================
# COST MATRIX
# =============================================================================

# Build the lifted cost matrix for output neuron j.
# Encodes: c(X) = tr(C X) = x_j (the j-th output neuron's value).
function get_cost(h, j)
    dim_matvar = sum(h) + 1
    C   = zeros(dim_matvar, dim_matvar)
    ind = sum(h[1:end-1]) + j  # flattened index of output neuron j

    C[ind, end] = 1/2
    C[end, ind] = 1/2

    return C
end

# =============================================================================
# CONSTRAINT BUILDERS
# =============================================================================

# Build lifted inequality constraints encoding the McCormick/RLT input bounds:
#   x_j² - (l+u) x_j + l·u ≤ 0  →  tr(A X) ≤ -l·u
function get_input_constraints(h, dim_matvar, l_bounds, u_bounds)
    all_ai = []
    all_bi = []

    for (i, h_i) in enumerate(h[1:1])  # input layer only
        lb_layer = l_bounds[i]
        ub_layer = u_bounds[i]

        for j in 1:h_i
            lb  = lb_layer[j]
            ub  = ub_layer[j]
            ind = sum(h[1:i-1]) + j  # flattened neuron index

            # Constraint: x² - (l+u)x + lu ≤ 0  →  tr(A X) ≤ -lu
            A = zeros(dim_matvar, dim_matvar)
            A[ind, ind]   =  1.0
            A[ind, end]   = -(lb + ub) / 2
            A[end, ind]   = -(lb + ub) / 2

            push!(all_ai, A)
            push!(all_bi, -lb * ub)
        end
    end

    return all_ai, all_bi
end

# Build lifted inequality and equality constraints encoding ReLU behaviour.
# For hidden neurons: x_{i+1} ≥ 0, x_{i+1} ≥ W x_i + b, and the
# complementarity equation x_{i+1}(x_{i+1} - W x_i - b) = 0.
# For the output layer: x_{out} = W x_{last_hidden} + b (linear equality).
function get_relu_constraints(h, dim_matvar, W, b)
    all_A_ineq = []
    all_b_ineq = []
    all_A_eq   = []
    all_b_eq   = []

    final_layer_index = length(h)

    for (i, h_i) in enumerate(h[1:end-1])
        h_next    = h[i+1]
        ind       = sum(h[1:i-1]) + 1      # first neuron index in layer i
        ind_next  = sum(h[1:i]) + 1        # first neuron index in layer i+1

        for j in 1:h_next
            out_ind = ind_next + j - 1
            is_hidden_next = (i + 1 != final_layer_index)

            if is_hidden_next
                # x_{i+1,j} ≥ 0
                A = zeros(dim_matvar, dim_matvar)
                A[out_ind, end] = -0.5
                A[end, out_ind] = -0.5
                push!(all_A_ineq, A)
                push!(all_b_ineq, 0.0)

                # x_{i+1,j} ≥ W[i][j,:] x_i + b[i][j]
                A = zeros(dim_matvar, dim_matvar)
                for k in 1:h_i
                    in_ind = ind + k - 1
                    w = W[i][j, k]
                    A[in_ind, end] +=  0.5 * w
                    A[end, in_ind] +=  0.5 * w
                end
                A[out_ind, end] += -0.5
                A[end, out_ind] += -0.5
                push!(all_A_ineq, A)
                push!(all_b_ineq, -b[i][j])

                # x_{i+1,j}(x_{i+1,j} - W[i][j,:] x_i - b[i][j]) = 0
                A = zeros(dim_matvar, dim_matvar)
                A[out_ind, out_ind] = 1.0
                for k in 1:h_i
                    in_ind = ind + k - 1
                    w = W[i][j, k]
                    A[out_ind, in_ind] -= 0.5 * w
                    A[in_ind, out_ind] -= 0.5 * w
                end
                A[out_ind, end] -= 0.5 * b[i][j]
                A[end, out_ind] -= 0.5 * b[i][j]
                push!(all_A_eq, A)
                push!(all_b_eq, 0.0)
            else
                # Output layer: x_{out,j} = W x_{last_hidden} + b (linear, no ReLU)
                A = zeros(dim_matvar, dim_matvar)
                for k in 1:h_i
                    in_ind = ind + k - 1
                    w = W[i][j, k]
                    A[in_ind, end] -= 0.5 * w
                    A[end, in_ind] -= 0.5 * w
                end
                A[out_ind, end] += 0.5
                A[end, out_ind] += 0.5
                push!(all_A_eq, A)
                push!(all_b_eq, b[i][j])
            end
        end
    end

    return all_A_ineq, all_b_ineq, all_A_eq, all_b_eq
end

# =============================================================================
# FEASIBLE POINT (network forward pass)
# =============================================================================

# Construct a feasible point for the SDP from a concrete network evaluation.
function get_feas_pt(x_sample, h, W, b, all_constraints_ai_eq, all_constraints_bi_eq, all_constraints_ai_ineq, all_constraints_bi_ineq)
    _, post_activ = forward_pass_with_states(x_sample, h, W, b)
    feas_pt = vcat(reduce(vcat, post_activ), 1)
    return feas_pt
end

# =============================================================================
# COST EVALUATION
# =============================================================================

function compute_cost(X, cost)
    return tr(cost * X)
end

# =============================================================================
# CONSTRAINT ASSEMBLY
# =============================================================================

# Assemble all SDP constraints, using IBP (interval bound propagation) to
# obtain tighter input bounds for the ReLU relaxation.
# Returns (ai_eq, bi_eq, ai_ineq, bi_ineq).
function get_sdp_constraints(n, h, W, b, l_in, u_in)
    dim_matvar = sum(h) + 1
    ai_ineq = []
    bi_ineq = []
    ai_eq   = []
    bi_eq   = []

    # IBP: propagate bounds through the network to tighten ReLU constraints
    _, post_activ_lb = SDP_RELAX.forward_pass_with_states(l_in, h, W, b)
    _, post_activ_ub = SDP_RELAX.forward_pass_with_states(u_in, h, W, b)

    # Input bound constraints
    A_in, b_in = get_input_constraints(h, dim_matvar, post_activ_lb, post_activ_ub)
    ai_ineq = vcat(ai_ineq, A_in)
    bi_ineq = vcat(bi_ineq, b_in)

    # ReLU constraints
    A_ineq_relu, b_ineq_relu, A_eq_relu, b_eq_relu = get_relu_constraints(h, dim_matvar, W, b)
    ai_ineq = vcat(ai_ineq, A_ineq_relu)
    bi_ineq = vcat(bi_ineq, b_ineq_relu)
    ai_eq   = vcat(ai_eq,   A_eq_relu)
    bi_eq   = vcat(bi_eq,   b_eq_relu)

    # Homogenization constraint: z_{end} = 1
    A_homog        = zeros(dim_matvar, dim_matvar)
    A_homog[end, end] = 1
    ai_eq = vcat(ai_eq, [A_homog])
    bi_eq = vcat(bi_eq, [1.0])

    return ai_eq, bi_eq, ai_ineq, bi_ineq
end

# =============================================================================
# SOLVER
# =============================================================================

# Solve the canonical (c-SDP) relaxation via Mosek interior-point.
# Returns (X_opt, solve_time, termination_status).
function sdp_relaxation_solve(which_bound, all_constraints_ai_ineq, all_constraints_bi_ineq, all_constraints_ai_eq, all_constraints_bi_eq, cost, logs_dir; X_start=nothing)
    model      = Model(Mosek.Optimizer)
    dim_matvar = size(cost, 1)

    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS",   1e-6)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS",   1e-6)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-10)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED",  1e-10)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_INFEAS",     1e-10)
    set_optimizer_attribute(model, "MSK_IPAR_INTPNT_MAX_ITERATIONS", 5000)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS",           1)
    set_optimizer_attribute(model, "MSK_IPAR_PRESOLVE_USE",          0)

    @variable(model, X[1:dim_matvar, 1:dim_matvar] in PSDCone())

    if X_start !== nothing && !isempty(X_start)
        for i in 1:dim_matvar, j in 1:dim_matvar
            set_start_value(X[i, j], X_start[i, j])
        end
        set_optimizer_attribute(model, "MSK_IPAR_INTPNT_STARTING_POINT", 1)
    end

    set_silent(model)

    constraint_data = []

    # Inequality constraints
    for (ind, A) in enumerate(all_constraints_ai_ineq)
        cref = @constraint(model, tr(A*X) <= all_constraints_bi_ineq[ind], base_name="ineq$ind")
        push!(constraint_data, (cref, A, all_constraints_bi_ineq[ind], :leq))
    end

    # Equality constraints
    for (ind, A) in enumerate(all_constraints_ai_eq)
        cref = @constraint(model, tr(A*X) == all_constraints_bi_eq[ind], base_name="eq$ind")
        push!(constraint_data, (cref, A, all_constraints_bi_eq[ind], :eq))
    end

    if which_bound == 1
        @objective(model, Min,  tr(X*cost))
    elseif which_bound == 2
        @objective(model, Min, -tr(X*cost))
    end

    optimize!(model)

    if !(termination_status(model) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED] ||
         primal_status(model) == MOI.FEASIBLE_POINT)
        println("SDP solver did not find a feasible solution.")
        return NaN, NaN, NaN
    end

    return value.(X), JuMP.solve_time(model), termination_status(model)
end

# =============================================================================
# UTILITIES
# =============================================================================

# Forward pass returning pre- and post-activation vectors at each layer.
# ReLU is applied at all hidden layers; the output layer is linear.
function forward_pass_with_states(x::Vector{Float64}, h::Vector{Int}, W::Vector{Matrix{Float64}}, b::Vector{Vector{Float64}})
    n_weight_layers = length(h) - 1

    @assert length(W) == n_weight_layers
    @assert length(b) == n_weight_layers

    pre_activations  = Vector{Vector{Float64}}(undef, n_weight_layers)
    post_activations = Vector{Vector{Float64}}(undef, length(h))

    post_activations[1] = x

    for i in 1:n_weight_layers
        z = W[i] * post_activations[i] + b[i]
        pre_activations[i]  = z
        post_activations[i+1] = (i < n_weight_layers) ? max.(z, 0.0) : z
    end

    return pre_activations, post_activations
end