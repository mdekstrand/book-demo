#!/bin/bash
#SBATCH --mail-user=michaelekstrand@boisestate.edu
#SBATCH --mail-type=END --mail-type=FAIL
#SBATCH -J cs538
#SRUN -J cs538

node=$(hostname)
echo "Running job on node $node" >&2
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate cs538

# Boise State's SLURM cluster has aggresive ulimits, even with larger job requests
# Reset those limits
ulimit -v unlimited
ulimit -u 2048
ulimit -n 4096

# Configure LensKit threading based on SLURM
cpus="$SLURM_CPUS_ON_NODE"

if [ -z "$LK_NUM_PROCS" -a -n "$cpus" ]; then
    # get proces from SLURM
    procs=$(expr $cpus / 2)
    if [ "$procs" = 0 ]; then
        procs=1
    fi
    echo "using $procs LK processes"
    export LK_NUM_PROCS=$procs
fi

if [ -n "$cpus" ]; then
    echo "using $cpus Numba threads"
    export NUMBA_NUM_THREADS="$cpus"
    export MKL_NUM_THREADS=1
fi

export MKL_THREADING_LAYER=tbb

# Finally run the code
exec "$@"
