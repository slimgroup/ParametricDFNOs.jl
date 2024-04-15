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

# # TODO: Add seperate file for evaluation on trained FNO and provide weights on repo
# # To train from a checkpoint
# filename = "ep=80_mt=4_mx=4_my=4_nblocks=4_nc_in=4_nc_lift=20_nc_mid=128_nc_out=1_nt=51_nx=64_ny=64_p=1.jld2"
# DFNO_2D.loadWeights!(θ, filename, "θ_save", partition)

DFNO_2D.train!(trainConfig, model, θ)

MPI.Finalize()
