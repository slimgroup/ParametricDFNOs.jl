using HDF5
using ParametricDFNOs.DFNO_3D

#### PERLMUTTER Data Loading ####

function read_perlmutter_data(path::String, modelConfig::ModelConfig, rank::Int; ntrain::Int=1000, nvalid::Int=100)

    n = ntrain + nvalid

    function read_x_tensor(file_name, key, indices)
        # indices for xyzn -> cxyzn where c=n=1 (t gets introduced and broadcasted later)
        data = nothing
        h5open(file_name, "r") do file
            dataset = file[key]
            data = dataset[indices[1:3]...]
        end
        return reshape(data, 1, (size(data)...), 1)
    end
    
    function read_y_tensor(file_name, key, indices)
        # indices for xyztn -> cxyztn where c=n=1
        data = zeros(modelConfig.dtype, map(range -> length(range), indices[1:4]))
        h5open(file_name, "r") do file
            times = file[key]
            for t in indices[4]
                data[:, :, :, t - indices[4][1] + 1] = file[times[t]][indices[1:3]...]
            end
        end
        return reshape(data, 1, (size(data)...), 1)
    end
    
    x_train = zeros(modelConfig.dtype, modelConfig.nc_in * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], ntrain)
    y_train = zeros(modelConfig.dtype, modelConfig.nc_out * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], ntrain)
    x_valid = zeros(modelConfig.dtype, modelConfig.nc_in * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], nvalid)
    y_valid = zeros(modelConfig.dtype, modelConfig.nc_out * modelConfig.nt * modelConfig.nx ÷ modelConfig.partition[1], modelConfig.ny * modelConfig.nz ÷ modelConfig.partition[2], nvalid)

    idx = 1

    for entry in readdir(path; join=true)
        try
            x_file = entry * "/inputs.jld2"
            y_file = entry * "/outputs.jld2"

            dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, 
                                            ntrain=1, 
                                            nvalid=0, 
                                            x_file=x_file,
                                            y_file=y_file,
                                            x_key="K",
                                            y_key="saturations")

            x, y, _, _ = DFNO_3D.loadDistData(dataConfig, 
            dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)

            if idx <= ntrain
                x_train[:,:,idx] = x[:,:,1]
                y_train[:,:,idx] = y[:,:,1]
            else
                x_valid[:,:,idx-ntrain] = x[:,:,1]
                y_valid[:,:,idx-ntrain] = y[:,:,1]
            end
            (rank == 0) && println("Loaded data sample no. $(idx) / $(n)")
            idx == n && break
            idx += 1
        catch e
            (rank == 0) && println("Failed to load data sample no. $(idx). Error: $e")
            continue
        end
    end

    return x_train, y_train, x_valid, y_valid
end

function print_filenames(path::String, start::Int=1001, finish::Int=1200)
    idx = 1
    for entry in readdir(path; join=true)
        if idx >= start && idx <= finish
            println("Sample $idx : ", entry)
        end
        idx += 1
    end
end
