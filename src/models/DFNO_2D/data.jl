## Francis Code x,y,t TODO: Make this dist
function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32},nt::Int,dt::Float32)
    tsample = [(i-1)*dt for i = 1:nt]
    return gen_grid(n, d, tsample)
end

function gen_grid(n::Tuple{Integer, Integer},d::Tuple{Float32, Float32},tsample::Vector{Float32})
    nt = length(tsample)
    grid = zeros(Float32,n[1],n[2],nt,3)
    for i = 1:nt     
        grid[:,:,i,1] = repeat(reshape(collect(range(d[1],stop=n[1]*d[1],length=n[1])), :, 1)',n[2])' # x
        grid[:,:,i,2] = repeat(reshape(collect(range(d[2],stop=n[2]*d[2],length=n[2])), 1, :),n[1])   # z
        grid[:,:,i,3] .= tsample[i]   # t
    end
    return grid
end

function perm_to_tensor(x_perm::AbstractMatrix{Float32},grid::Array{Float32,4},AN::ActNorm)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny, nt, _ = size(grid)
    return cat(reshape(cat([AN(reshape(x_perm, nx, ny, 1, 1))[:,:,1,1] for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
    reshape(grid, nx, ny, nt, 3, 1), dims=4)
end

perm_to_tensor(x_perm::AbstractArray{Float32,3},grid::Array{Float32,4},AN::ActNorm) = cat([perm_to_tensor(x_perm[:,:,i],grid,AN) for i = 1:size(x_perm,3)]..., dims=5)
## End Francis Code

# Returns training and validation data in format cxytn distributed according to partition. TODO: Make this distributed
function loadData(partition; comm=MPI.COMM_WORLD)
    
    # TODO: Make this global similar to plot_path
    mkpath(datadir(model_name))

    perm_path = datadir(model_name, "perm_gridspacing15.0.mat")
    conc_path = datadir(model_name, "conc_gridspacing15.0.mat")

    rank = MPI.Comm_rank(comm)

    if rank == 0
        if ~isfile(perm_path)
            run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
                'perm_gridspacing15.0.mat -q -O $perm_path`)
        end
        if ~isfile(conc_path)
            run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
                'conc_gridspacing15.0.mat -q -O $conc_path`)
        end
    end

    MPI.Barrier(comm)

    perm = matread(perm_path)["perm"];

    ntrain = 1000
    nvalid = 100

    n = (64,64)   
    d = (1f0/64, 1f0/64)

    s = 1

    nt = 51
    dt = 1f0/(nt-1)

    AN = ActNorm(ntrain)
    AN.forward(reshape(perm[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

    grid = gen_grid(n, d, nt, dt)

    # Following Errors on Machine @ CODA Out of memory SIGKILL 9

    x_train = permutedims(perm_to_tensor(perm[1:s:end,1:s:end,1:ntrain],grid,AN), [4,1,2,3,5]);
    x_valid = permutedims(perm_to_tensor(perm[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN), [4,1,2,3,5]);

    perm = nothing #  Free the variable for now, TODO: Dist read
    conc = matread(conc_path)["conc"];

    y_train = permutedims(conc[1:nt,1:s:end,1:s:end,1:ntrain],[2,3,1,4]);
    y_valid = permutedims(conc[1:nt,1:s:end,1:s:end,ntrain+1:ntrain+nvalid],[2,3,1,4]);

    y_train = reshape(y_train, 1, (size(y_train)...))
    y_valid = reshape(y_valid, 1, (size(y_valid)...))

    # TODO: Introduce a new operator for future use
    x_train = UTILS.dist_tensor(x_train, size(x_train), [partition..., 1])
    y_train = UTILS.dist_tensor(y_train, size(y_train), [partition..., 1])
    x_valid = UTILS.dist_tensor(x_valid, size(x_valid), [partition..., 1])
    y_valid = UTILS.dist_tensor(y_valid, size(y_valid), [partition..., 1])

    return x_train, y_train, x_valid, y_valid
end
