using ParametricOperators
using ParametricOperators: ⊠
using Zygote
using Flux
using LinearAlgebra

using PyPlot
using Combinatorics

function save_scaling_graph(parameter_count, inf_timing, gradient_timing, filename::String; dpi::Int=300)
    # Scale parameter count to millions (M)
    parameter_count_millions = parameter_count ./ 1_000_000
    
    # Create the plot
    figure()
    plot(parameter_count_millions, inf_timing, marker="o", linestyle="-", color="blue", label="Inference Time")
    plot(parameter_count_millions, gradient_timing, marker="o", linestyle="-", color="red", label="Gradient Time")
    ylabel("Time (s)")
    xlabel("Parameter Count (M)")
    title("Parameters vs Time")
    
    # Add a legend to differentiate the lines
    legend()
    
    # Save to disk with specified DPI
    savefig(filename, dpi=dpi)
    close()  # Close the figure to free up memory
end


T = Complex{Float32}

nc_lift = 32
mx = 20
my = 20 
mt = 20 

mc_values = [8, 16, nc_lift] # [8, 16, mc]
mx_values = [5, 10, mx] # [16, 32, mt]
my_values = [5, 10, my] # [16, 32, mx]
mt_values = [5, 10, mt] # [16, 32, my]

function get_pcount(𝜃)
    total = 0
    for value in values(𝜃)
        total = total + prod(size(value))
    end
    return total
end

parameter_count = []
inference = []
gradients = []

settings = [[mc_val, mc_val, mt_val, mx_val, my_val] for mc_val in mc_values, mt_val in mt_values, mx_val in mx_values, my_val in my_values]

for (idx, setting) in enumerate(settings)
    println("Setting $idx / $(length(settings))")
    nc_lift, _, mt, mx, my = setting

    input_shape = (nc_lift, mt, mx*my)
    weight_shape = (nc_lift, nc_lift, mt, mx*my)

    factorization_ranks = [4, 4, 4, 4]

    G = ParMatrix(T, factorization_ranks[1], prod(factorization_ranks[2:end]))

    Uo = ParMatrix(T, weight_shape[1], factorization_ranks[1])
    UiT = ParMatrix(T, factorization_ranks[2], weight_shape[2])
    UtT = ParMatrix(T, factorization_ranks[3], weight_shape[3])
    UmT = ParMatrix(T, factorization_ranks[4], weight_shape[4])

    Ut = reduce(⊠, [UtT[:, j] for j in 1:weight_shape[3]])
    Um = reduce(⊠, [UmT[:, j] for j in 1:weight_shape[4]])

    I = ParIdentity(T, prod(input_shape[2:end]))

    O = (I ⊗ Uo) * (I ⊗ G) * (Um ⊗ Ut ⊗ UiT)

    x = rand(T, input_shape...)
    θ = init(O)

    inf_time_taken = @timed O(θ) * vec(x)
    time_taken = @timed Zygote.gradient(θ -> norm(O(θ) * vec(x)), θ)[1]

    inf_time_taken = @timed O(θ) * vec(x)
    time_taken = @timed Zygote.gradient(θ -> norm(O(θ) * vec(x)), θ)[1]

    push!(inference, inf_time_taken[2])
    push!(gradients, time_taken[2])

    push!(parameter_count, get_pcount(θ)*10000)
end

sorted_indices = sortperm(parameter_count)
sorted_parameter_count = parameter_count[sorted_indices]
sorted_inference = inference[sorted_indices]
sorted_gradient = gradients[sorted_indices]

# Filter out values where either inference or gradient time is greater than 0.1
filtered_parameter_count = []
filtered_inference = []
filtered_gradient = []

for i in 1:81
    if sorted_gradient[i] <= 0.06
        push!(filtered_parameter_count, sorted_parameter_count[i])
        push!(filtered_inference, sorted_inference[i])
        push!(filtered_gradient, sorted_gradient[i])
    end
end
