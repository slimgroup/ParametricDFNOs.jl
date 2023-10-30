## Log into NERSC

```
ssh -l <username> -i ~/.ssh/nersc perlmutter.nersc.gov
```

## ENV Setup

### 1. Install Julia 1.8.5

```
> module load julia/1.6.0
> julia
julia> ]add UpdateJulia
julia> using UpdateJulia
julia> update_julia("1.8.5", systemwide=false)
```

### 2. Clone dfno

```
git clone https://github.com/turquoisedragon2926/dfno.git
```

### 3. Setup dfno env

```
> cd dfno
> julia-1.8
julia> ]
(v1.8) pkg> activate ./
(dfno) pkg> instantiate
```

### 4. Setup mpiexecjl

```
> julia-1.8
julia> using MPI
julia> MPI.install_mpiexecjl()
julia> exit()
```

## Submitting jobs

- Open up `dfno/examples/permutter/train.jl`
- Modify `dim=20` to one of [20, 40, 80, 160]
    - The dataset would need to exist at `/global/cfs/projectdirs/m3863/mark/training-data/training-samples/v5/$(dim)³`
- Run `rank == 0 && DFNO_3D.print_storage_complexity(modelConfig, batch=2)` to get an idea of required compute for given model
- Modify `dfno/examples/permutter/train.sh` with required resources and flags
- Modify `partition=[1,4,4,4,1]` in `dfno/examples/permutter/train.jl`
- Submit job using `sbatch examples/permutter/train.sh`
