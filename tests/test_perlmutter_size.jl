using Pkg
Pkg.activate("./")

using HDF5
using MPI

sizes = [20, 40, 80, 160]
t_size = 55

for size in sizes
    dataset_path = "/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5/$(size)³"
    for entry in readdir(dataset_path; join=true)

        perm_file = entry * "/inputs.jld2"
        conc_file = entry * "/outputs.jld2"

        h5open(perm_file, "r") do file
            dims = file["K"]
            for t in size(dims)
                t !== size && println(" Problem @ $(perm_file)")
            end
        end

        h5open(conc_file, "r") do file
            times = file["saturations"]
            length(times) !== t_size && println(" Problem with time @ $(conc_file)")
        end
    end
end
