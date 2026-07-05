# This script sets environment variables for improved CPU performance
# This is not required to run, but can give a noticeable benefit

# These flags are the most critical
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# This flag sometimes makes a difference when factorizing matrices
export OPENBLAS_NUM_THREADS=1

# These flags are less important (from what I've found)
# but they are often mentioned in various threads on the Jax github
# Leaving them commented out for now
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
