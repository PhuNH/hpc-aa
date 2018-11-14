CC		= nvcc
EXEC	= out

SOURCES = \
cuda_mmult_kernels.cu \
cuda_mmult.cu

OBJS = $(SOURCES:.cu=.o)

%.o: %.cu
	$(CC) -c -O3 -Wno-deprecated-gpu-targets -o $@ $<

all: $(OBJS)
	$(CC) -link -L/usr/local/cuda/lib64/ -O3 -Wno-deprecated-gpu-targets $(OBJS) -o $(EXEC)

clean:
	@rm -f $(OBJS) $(EXEC)
