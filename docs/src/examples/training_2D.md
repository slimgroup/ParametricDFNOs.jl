### Training 2D Time varying FNO

!!! note "Jump right in"
    To get started, you can run some [examples](https://github.com/turquoisedragon2926/ParametricDFNOs.jl-Examples)

Make sure to add necessary dependencies. You might also need to load a proper MPI implementation based on your hardware.

```julia
julia> ]
(v1.9) activate /path/to/your/environment 
(venv) add JLD2 FileIO MAT MPI CUDA ParametricDFNOs
```

!!! warning "To run on multiple GPUs"
    If you wish to run on multiple GPUs, make sure the GPUs are binded to different tasks. The approach we use is to unbind our GPUs on request and assign manually:

    ```julia
    CUDA.device!(rank % 4)
    ```

    which might be different if you have more or less than 4 GPUs per node. Also, make sure your MPI distribution is functional.

```julia
using MPI
using CUDA
using ParametricDFNOs.DFNO_2D
using ParametricDFNOs.UTILS
using JLD2, FileIO, MAT

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

global gpu_flag = parse(Bool, get(ENV, "DFNO_2D_GPU", "0"))
DFNO_2D.set_gpu_flag(gpu_flag)

# Julia requires you to manually assign the gpus, modify to your case.
DFNO_2D.gpu_flag && (CUDA.device!(rank % 4))
partition = [1, pe_count]

modelConfig = DFNO_2D.ModelConfig(nblocks=4, partition=partition)

### Setup example dataset ###

perm_path_mat = "data/DFNO_2D/perm_gridspacing15.0.mat"
conc_path_mat = "data/DFNO_2D/conc_gridspacing15.0.mat"
perm_store_path_jld2 = "data/DFNO_2D/perm_gridspacing15.0.jld2"
conc_store_path_jld2 = "data/DFNO_2D/conc_gridspacing15.0.jld2"

# TODO: Host a .jld2 file with correct dimensions
# Check if .jld2 files already exist and skip processing if they do
if isfile(perm_store_path_jld2) && isfile(conc_store_path_jld2)
    rank == 0 && println("JLD2 files already exist, skipping processing.")
elseif rank == 0
    ensure_directory = path -> isdir(path) || mkpath(path)
    ensure_downloaded = (url, path) -> isfile(path) || run(`wget $url -q -O $path`)

    # Ensure necessary directories exist
    ensure_directory(dirname(perm_path_mat))
    ensure_directory(dirname(perm_store_path_jld2))
    
    # Ensure .mat files are downloaded
    ensure_downloaded("https://www.dropbox.com/s/o35wvnlnkca9r8k/perm_gridspacing15.0.mat", perm_path_mat)
    ensure_downloaded("https://www.dropbox.com/s/mzi0xgr0z3l553a/conc_gridspacing15.0.mat", conc_path_mat)

    # Load .mat files
    perm = matread(perm_path_mat)["perm"];
    conc = matread(conc_path_mat)["conc"];

    conc = permutedims(conc, [2, 3, 1, 4])

    # Save data to .jld2 format
    @save perm_store_path_jld2 perm
    @save conc_store_path_jld2 conc
end

MPI.Barrier(comm)

#############################

dataConfig = DFNO_2D.DataConfig(modelConfig=modelConfig, 
                                x_key = "perm",
                                x_file = perm_store_path_jld2,
                                y_key="conc",
                                y_file=conc_store_path_jld2)

x_train, y_train, x_valid, y_valid = DFNO_2D.loadDistData(dataConfig)

trainConfig = DFNO_2D.TrainConfig(
    epochs=10,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    plot_every=1
)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
```

If you have [`mpiexecjl`](https://juliaparallel.org/MPI.jl/stable/usage/#Installation) set up, you can run the above by doing:

```shell
mpiexecjl --project=/path/to/your/environment -n NTASKS julia code_above.jl
```

OR if you have a HPC cluster with [`slurm`](https://slurm.schedmd.com/documentation.html) set up, you can do:

```shell
salloc --gpus=NTASKS --time=01:00:00 --ntasks=NTASKS --gpus-per-task=1 --gpu-bind=none
srun julia --project=/path/to/your/environment code_above.jl
```

!!! warning "Allocation"
    Your `salloc` might look different based on your HPC cluster
