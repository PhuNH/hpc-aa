### CMakeLists.txt for Sparse
project(sparse_matrix)

cmake_minimum_required(VERSION 2.8)

# Find CUDA
find_package(CUDA REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -Wno-deprecated-gpu-targets
)

# Create the Compilation Target
cuda_add_executable(
    sparse
    main.c
    mmio.c
    kernels.cu
    pagerank.c
    poisson.c
    gnuplot_i.c
)

# Link all necessary libraries
target_link_libraries(
    sparse
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
)

# Add CUBLAS to target
CUDA_ADD_CUBLAS_TO_TARGET(sparse)
