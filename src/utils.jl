module UTILS

using MPI
using HDF5
using ParametricOperators

cpu = ParametricOperators.cpu

function unique_range(ranges)
    unique_ranges = unique(vcat(ranges...))
    return isempty(unique_ranges) ? [1:1] : ranges
end

function collect_dist_tensor(local_tensor, global_shape, partition, parent_comm)
    comm_cart = MPI.Cart_create(parent_comm, partition)
    coords = MPI.Cart_coords(comm_cart)

    sparse = zeros(eltype(local_tensor), global_shape...)
    indexes = _get_local_indices(global_shape, partition, coords)

    sparse[indexes...] = local_tensor
    return MPI.Reduce(sparse, MPI.SUM, 0, parent_comm)
end

"""
    dist_loss(local_pred_y, local_true_y)

Calculates the distributed normalized root mean squared error (NRMSE) between predicted and actual values.

# Arguments
- `local_pred_y`: Local tensor of predicted values, typically a subset of the whole prediction corresponding to the data handled by a specific node.
- `local_true_y`: Local tensor of actual values corresponding to a subset of the data

# Returns
- A scalar value representing the relative L2 error of the predictions to the true values, which quantifies the prediction error normalized by the magnitude of the actual values.
"""
function dist_loss(local_pred_y, local_true_y)
    s = sum((vec(local_pred_y) - vec(local_true_y)) .^ 2)

    reduce_norm = ParReduce(eltype(local_pred_y))
    reduce_y = ParReduce(eltype(local_true_y))

    norm_diff = √(reduce_norm([s])[1])
    norm_y = √(reduce_y([sum(local_true_y .^ 2)])[1])

    return norm_diff / norm_y
end

function dist_sum(local_vec)
    reduce_sum = ParReduce(eltype(local_vec))
    return sum(reduce_sum(local_vec))
end

function _get_local_indices(global_shape, partition, coords)
    indexes = []
    for (dim, value) in enumerate(global_shape)
        local_size = value ÷ partition[dim]
        start = 1 + coords[dim] * local_size

        r = value % partition[dim]

        if coords[dim] < r
            local_size += 1
            start += coords[dim]
        else
            start += r
        end

        push!(indexes, start:start+local_size-1)
    end
    return indexes
end

function dist_tensor(tensor, global_shape, partition; parent_comm=MPI.COMM_WORLD)
    comm_cart = MPI.Cart_create(parent_comm, partition)
    coords = MPI.Cart_coords(comm_cart)

    indexes = _get_local_indices(global_shape, partition, coords)
    tensor = reshape(tensor, global_shape)
    return tensor[indexes...]
end

"""
    dist_read_tensor(file_name, key, indices)

Reads a tensor slice from an HDF5 file based on provided indices and reshapes the result.

# Arguments
- `file_name`: The name or path of the HDF5 file from which data is to be read.
- `key`: The key within the HDF5 file that corresponds to the dataset of interest.
- `indices`: The indices specifying the slice of the dataset to be extracted.

# Returns
- A reshaped array where the first dimension is singleton, extending the dimensions of the original data slice by one.

# Functionality
- Opens an HDF5 file in read-only mode.
- Accesses the dataset associated with the provided `key`.
- Extracts the slice of the dataset specified by `indices`.
- Reshapes the extracted data, adding a singleton dimension at the beginning.

# Example Usage
```julia
# To read a specific slice from a dataset within an HDF5 file
tensor_data = dist_read_tensor("data.h5", "dataset_key", (1:10, 5:15, 2:2))
```
"""
function dist_read_tensor(file_name, key, indices)
    data = nothing
    h5open(file_name, "r") do file
        dataset = file[key]
        data = dataset[indices...]
    end
    return reshape(data, 1, (size(data)...))
end

function get_dist_indices(total_size, total_workers, coord)
    # Calculate the base size each worker will handle
    base_size = div(total_size, total_workers)
    
    # Calculate the number of workers that will handle an extra element
    extras = total_size % total_workers
    
    # Determine the start and end indices for the worker
    if coord < extras
        start_index = coord * (base_size + 1) + 1
        end_index = start_index + base_size
    else
        start_index = coord * base_size + extras + 1
        end_index = start_index + base_size - 1
    end

    return start_index, end_index
end

export dist_loss, collect_dist_tensor, dist_tensor, dist_read_tensor, get_dist_indices, dist_sum, unique_range

end
