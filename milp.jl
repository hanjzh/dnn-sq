using NeuralVerification
using LazySets
using MIPVerify
using GLPK
using JuMP

# Solve the neural network verification problem as a MILP using MIPVerify.
# Returns (optimal_output_value, full_activation_vector_homogenized, solve_time, status).
function solve_milp(which_bound, nnet_name, output_index, l_in, u_in)
    nnet      = read_nnet(nnet_name)
    input_set = Hyperrectangle(low=l_in, high=u_in)
    problem   = NeuralVerification.Problem(nnet, input_set, Nothing)

    solver = NeuralVerification.MIPVerify()
    model  = NeuralVerification.Model(solver.optimizer)

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    δ = NeuralVerification.init_vars(model, problem.network, :δ, binary=true)

    model[:bounds]     = NeuralVerification.get_bounds(problem, before_act=true)
    model[:before_act] = true

    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BoundedMixedIntegerLP())

    if which_bound == 1
        JuMP.@objective(model, Min, last(z)[output_index])
    elseif which_bound == 2
        JuMP.@objective(model, Max, last(z)[output_index])
    end

    JuMP.optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        full_vector = extract_full_vector(z)
        return value(last(z)[output_index]), vcat(full_vector, 1.0), JuMP.solve_time(model), termination_status(model)
    else
        println("MILP failed. Status: ", termination_status(model))
        return NaN, nothing, JuMP.solve_time(model), termination_status(model)
    end
end

# Flatten the layered neuron activation variables z into a single vector.
function extract_full_vector(z)
    return reduce(vcat, [value.(layer_vars) for layer_vars in z])
end