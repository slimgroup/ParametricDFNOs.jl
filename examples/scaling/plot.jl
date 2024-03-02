using Pkg
Pkg.activate("./")

using JLD2
using PyPlot

# Function to read the specified keys from a JLD2 file
function read_keys_from_jld2(file_path, keys)
    data = Dict()
    jldopen(file_path, "r") do file
        for key in keys
            data[key] = read(file, key)
        end
    end
    return data
end

folder = ARGS[1]
directory = "examples/scaling/$folder/"

# Get all JLD2 files in the current directory
jld2_files = filter(x -> occursin(r"\.jld2$", x), readdir(directory))

# Initialize arrays to hold the data
data_points = []

# Read the data from each file
for file in jld2_files
    full_path = joinpath(directory, file)
    data = read_keys_from_jld2(full_path, ["y_time", "gpus", "nodes", "dimx", "dimy", "dimz", "grads_time"])
    push!(data_points, (data["gpus"], data["y_time"], data["dimx"], data["dimy"], data["dimz"], data["nodes"], data["grads_time"]))
end

# Sort the data by the 'gpus' value
sorted_data = sort(data_points, by=x->x[1])

# Extract the sorted data
gpus = [point[1] for point in sorted_data]
y_times = [point[2] for point in sorted_data]
dims = ["($(point[3]), $(point[4]), $(point[5]))" for point in sorted_data]
nodes = [point[6] for point in sorted_data]
grads_times = [point[7] for point in sorted_data]

# Now plot the data
fig, ax1 = subplots()
ax1.plot(1:length(gpus), y_times, "s-b", label="Inference Time")  # Plot against evenly spaced x-axis
ax1.plot(1:length(gpus), grads_times, "o-r", label="Gradient Time")  # Add this line for gradients time

# # Add a grey dotted line for comparison
# # Assuming you want a line with a slope of 1 for comparison
# max_val = max(maximum(y_times), length(gpus))  # Find the maximum value for x and y
# ax1.plot(1:max_val, 1:max_val, "--", color="grey", label="Ideal Scaling")  # Grey dotted line

# Calculate padding for the y-axis
y_padding = (maximum(grads_times) - minimum(y_times)) * 0.20  # 5% padding from the range of grads_times and y_times
ax1.set_ylim(minimum(y_times) - y_padding, maximum(grads_times) + y_padding)

ax1.set_xlabel("# GPUs/Nodes", labelpad=10, fontsize=20)
ax1.set_ylabel("Runtime [s]", fontsize=20)
ax1.set_title("Weak Scaling", pad=10, fontsize=20)
ax1.legend()
ax1.grid(true)

# Set the x-axis ticks to be evenly spaced and label them with GPUs and Nodes
gpu_node_labels = ["$(gpus[i])/$(nodes[i])" for i in 1:length(gpus)]
ax1.set_xticks(1:length(gpus))  # Evenly space the ticks
ax1.set_xticklabels(gpu_node_labels)  # Label the ticks with the combined GPU/Node values
ax1.legend(fontsize=20)

# Create a second x-axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Ensure the new x-axis has the same limits as the original
ax2.set_xticks(1:length(gpus))  # Set the ticks to be at the same evenly spaced positions
ax2.set_xticklabels(dims, rotation=30) # Set the labels to be the dimension information
# ax2.set_xlabel("Dimensions (dimx, dimy, dimz)")

# Increase the size of the axis tick labels
ax1.tick_params(axis="x", labelsize=20)
ax2.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)

# Adjust the layout to make sure everything fits
fig.tight_layout()

# Show the plot with both x-axes
show()
