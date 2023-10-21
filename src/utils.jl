module UTILS

using MPI
using ParametricOperators

function collect_dist_tensor(local_tensor, global_shape, partition, parent_comm)
    comm_cart = MPI.Cart_create(parent_comm, partition)
    coords = MPI.Cart_coords(comm_cart)

    sparse = zeros(eltype(local_tensor), global_shape...)
    indexes = _get_local_indices(global_shape, partition, coords)

    sparse[indexes...] = local_tensor
    return MPI.Reduce(sparse, MPI.SUM, 0, parent_comm)
end

function dist_loss(local_pred_y, local_true_y)
    s = sum((vec(local_pred_y) - vec(local_true_y)) .^ 2)

    reduce_norm = ParReduce(eltype(local_pred_y))
    reduce_y = ParReduce(eltype(local_true_y))

    norm_diff = √(reduce_norm([s])[1])
    norm_y = √(reduce_y([sum(local_true_y .^ 2)])[1])

    return norm_diff / norm_y
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

export dist_loss, collect_dist_tensor, dist_tensor

end
